# Comprehensive Python Interview Questions 

## Advanced Python Features

### 1. Question: Explain Python's descriptor protocol and provide a practical example where you've used it (or would use it) in a production environment.

**Answer:**

Descriptors are Python objects that implement at least one of the following methods: `__get__`, `__set__`, or `__delete__`. They enable controlled access to attributes and are the underlying magic behind many Python features like properties, classmethod, and staticmethod.

A descriptor that implements only `__get__` is a non-data descriptor, while one that implements `__set__` or `__delete__` is a data descriptor, which takes precedence over instance dictionaries.

Here's a real-world example implementing a type-validated field system:

```python
class Field:
    def __init__(self, field_type, required=False, default=None, validators=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.validators = validators or []
        self.name = None  # Will be set by ModelMeta
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self.default)
    
    def __set__(self, instance, value):
        if value is None:
            if self.required:
                raise ValueError(f"{self.name} is required")
            instance.__dict__[self.name] = None
            return
            
        if not isinstance(value, self.field_type):
            raise TypeError(f"{self.name} must be of type {self.field_type.__name__}")
        
        for validator in self.validators:
            validator(value)
            
        instance.__dict__[self.name] = value

class ModelMeta(type):
    def __new__(mcs, name, bases, namespace):
        for key, value in namespace.items():
            if isinstance(value, Field):
                value.__set_name__(None, key)
        return super().__new__(mcs, name, bases, namespace)

class Model(metaclass=ModelMeta):
    pass

# Usage in a real application - API request validation
def min_length(length):
    def validate(value):
        if len(value) < length:
            raise ValueError(f"Value must be at least {length} characters")
    return validate

class User(Model):
    username = Field(str, required=True, validators=[min_length(3)])
    email = Field(str, required=True)
    age = Field(int, required=False, default=0)
    
    def __repr__(self):
        return f"User(username='{self.username}', email='{self.email}', age={self.age})"

# In production code, for API request validation:
def create_user_endpoint(request_data):
    try:
        user = User()
        user.username = request_data.get('username')
        user.email = request_data.get('email')
        user.age = request_data.get('age')
        # Save to database...
        return {"status": "success", "user": repr(user)}
    except (ValueError, TypeError) as e:
        return {"status": "error", "message": str(e)}
```

This approach provides several benefits:

1. **Centralized validation** logic
2. **Type safety** without runtime overhead of constant checks
3. **Self-documenting code** - the requirements for each field are visible in the model definition
4. **Reusability** across multiple models

I've used this pattern in a RESTful API service to validate incoming JSON data before processing, reducing boilerplate and ensuring consistent validation.

### 2. Question: Explain how context managers work in Python and how you would implement a custom context manager for database connections.

**Answer:**

Context managers in Python are objects designed to work with the `with` statement, providing resource acquisition and release. They implement the context management protocol via `__enter__` and `__exit__` methods.

There are two ways to create context managers:

1. Class-based implementation with `__enter__` and `__exit__` methods
2. Using the `@contextmanager` decorator from `contextlib`

Here's a production-ready database connection context manager:

```python
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    def __init__(self, host, database, user, password, min_conn=1, max_conn=10):
        self.pool = ThreadedConnectionPool(
            min_conn, 
            max_conn,
            host=host,
            database=database,
            user=user,
            password=password
        )
        logger.info(f"Initialized connection pool to {database} on {host}")
        
    def __enter__(self):
        self.conn = self.pool.getconn()
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        return self.cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Database error: {exc_type.__name__}: {exc_val}")
            self.conn.rollback()
        else:
            self.conn.commit()
        
        self.cursor.close()
        self.pool.putconn(self.conn)
        logger.debug("Released connection back to pool")
        
        # Don't suppress exception
        return False
    
    def close(self):
        """Close all connections in the pool"""
        self.pool.closeall()
        logger.info("Closed all database connections")

# Alternative implementation using the contextmanager decorator
@contextmanager
def db_transaction(db_manager):
    conn = db_manager.pool.getconn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction error: {e}")
        raise
    finally:
        cursor.close()
        db_manager.pool.putconn(conn)

# Usage in production code
DB_CONFIG = {
    'host': 'localhost',
    'database': 'myapp',
    'user': 'postgres',
    'password': 'secret',
    'min_conn': 2,
    'max_conn': 20
}

db_manager = DatabaseConnectionManager(**DB_CONFIG)

# Class-based approach
def get_user(user_id):
    with db_manager as cursor:
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        return cursor.fetchone()

# Decorator-based approach
def get_products(category, limit=10):
    with db_transaction(db_manager) as cursor:
        cursor.execute(
            "SELECT * FROM products WHERE category = %s LIMIT %s", 
            (category, limit)
        )
        return cursor.fetchall()

# Cleanup when application shuts down
import atexit
atexit.register(db_manager.close)
```

The key benefits this provides in a production system:

1. **Connection pooling** for better performance and resource utilization
2. **Automatic transaction management** - commits on success, rollbacks on exception
3. **Resource cleanup** even in the event of exceptions
4. **Simplified client code** without explicit connection management
5. **Centralized logging** of database operations and errors

I've used this pattern in high-load microservices to ensure proper database connection lifecycle management and prevent connection leaks, which were previously causing our application to run out of available connections during traffic spikes.

### 3. Question: How does asyncio work in Python, and how would you design a high-throughput web scraper using asyncio?

**Answer:**

Asyncio is Python's built-in framework for writing concurrent code using the async/await syntax. It uses an event loop to manage and schedule asynchronous tasks, allowing programs to perform I/O operations concurrently without using threads or processes.

The key components are:

- **Event loop**: Manages and schedules tasks
- **Coroutines**: Functions defined with `async def` that can be paused and resumed
- **Tasks**: Wrappers around coroutines to track their execution
- **Futures**: Represent the result of a computation that may not have completed yet

Here's a production-grade web scraper using asyncio, with rate limiting, error handling, and proper resource management:

```python
import asyncio
import aiohttp
import logging
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import csv
from aiolimiter import AsyncLimiter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncWebScraper:
    def __init__(self, concurrency=10, rate_limit=3, timeout=30):
        """
        Initialize the web scraper with configuration options
        
        Args:
            concurrency: Maximum number of concurrent requests
            rate_limit: Maximum requests per second per domain
            timeout: Request timeout in seconds
        """
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.rate_limiters = {}  # Domain-specific rate limiters
        self.rate_limit = rate_limit
        self.results = []
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _get_domain(self, url):
        """Extract domain from URL for domain-specific rate limiting"""
        return urlparse(url).netloc
    
    def _get_limiter(self, domain):
        """Get or create a rate limiter for the domain"""
        if domain not in self.rate_limiters:
            self.rate_limiters[domain] = AsyncLimiter(self.rate_limit, 1)
        return self.rate_limiters[domain]
    
    async def fetch_page(self, url, headers=None):
        """Fetch a single page with rate limiting and error handling"""
        domain = self._get_domain(url)
        limiter = self._get_limiter(domain)
        
        headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with self.semaphore:
            logger.info(f"Fetching URL: {url}")
            async with limiter:
                try:
                    start_time = time.time()
                    async with self.session.get(url, headers=headers) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                            return None
                        
                        content = await response.text()
                        elapsed = time.time() - start_time
                        logger.info(f"Fetched {url} in {elapsed:.2f} seconds")
                        return content
                except asyncio.TimeoutError:
                    logger.error(f"Timeout while fetching {url}")
                    return None
                except Exception as e:
                    logger.error(f"Error fetching {url}: {str(e)}")
                    return None
    
    async def parse_page(self, url, parser_func):
        """Fetch and parse a page using the provided parser function"""
        content = await self.fetch_page(url)
        if not content:
            return None
        
        try:
            result = parser_func(content, url)
            if result:
                self.results.append(result)
            return result
        except Exception as e:
            logger.error(f"Error parsing {url}: {str(e)}")
            return None
    
    async def scrape_urls(self, urls, parser_func):
        """Scrape multiple URLs concurrently"""
        tasks = [self.parse_page(url, parser_func) for url in urls]
        return await asyncio.gather(*tasks)
    
    def save_results_csv(self, filename, fieldnames):
        """Save results to a CSV file"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                if result:
                    writer.writerow(result)
        logger.info(f"Saved {len(self.results)} results to {filename}")

# Example parser function
def parse_product_page(content, url):
    soup = BeautifulSoup(content, 'html.parser')
    try:
        return {
            'url': url,
            'title': soup.select_one('h1.product-title').text.strip(),
            'price': soup.select_one('span.price').text.strip(),
            'description': soup.select_one('div.description').text.strip(),
            'rating': soup.select_one('div.rating').get('data-rating', 'N/A')
        }
    except AttributeError:
        logger.warning(f"Failed to parse product data from {url}")
        return None

# Example usage
async def main():
    urls = [
        "https://example.com/product/1",
        "https://example.com/product/2",
        # Add more URLs...
    ]
    
    async with AsyncWebScraper(concurrency=5, rate_limit=2) as scraper:
        await scraper.scrape_urls(urls, parse_product_page)
        scraper.save_results_csv('products.csv', 
                                ['url', 'title', 'price', 'description', 'rating'])

if __name__ == "__main__":
    asyncio.run(main())
```

Key design considerations for a high-throughput production scraper:

1. **Resource Management**
    
    - Use `async with` (context managers) for proper cleanup
    - Implement connection pooling via shared aiohttp session
    - Limit concurrency with semaphores to avoid overwhelming resources
2. **Rate Limiting**
    
    - Domain-specific rate limiting to respect servers
    - Configurable limits based on target site requirements
3. **Error Handling**
    
    - Graceful handling of network errors, timeouts, and parsing errors
    - Comprehensive logging for monitoring and troubleshooting
4. **Performance Optimizations**
    
    - Concurrent requests to maximize throughput
    - Timeout configuration to avoid hanging tasks
5. **Respectful Web Citizenship**
    
    - Proper user agent headers
    - Rate limiting to avoid putting undue strain on target sites

This design enables efficient web scraping while being considerate of resources and server load. In production, I've used similar implementations to collect product information from e-commerce sites and gather research data from academic sources.

## System Design & Architecture

### 4. Question: How would you design a highly scalable notification system using Python that can handle multiple communication channels (email, SMS, push notifications) with potential delays in third-party services?

**Answer:**

A scalable notification system needs to handle high throughput, work asynchronously, and be resilient to failures. Here's a comprehensive design:

**Architecture Overview:**

1. **API Layer**: Receives notification requests
2. **Queue System**: Decouples notification processing from request handling
3. **Worker Pool**: Processes notifications from queues
4. **Channel Adapters**: Handles specific delivery methods
5. **Storage Layer**: Tracks notification status and history
6. **Retry Mechanism**: Handles failed deliveries
7. **Monitoring System**: Tracks performance and failures

**Implementation Components:**

```python
# core/models.py
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import uuid

class NotificationChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"

class NotificationStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"

class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class Notification(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    channel: NotificationChannel
    recipient: str
    subject: Optional[str] = None
    content: Union[str, Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    priority: NotificationPriority = NotificationPriority.NORMAL
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    scheduled_for: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

# api/router.py
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from core.models import Notification, NotificationChannel
from services.notification_service import NotificationService

router = APIRouter()

@router.post("/notifications", status_code=202)
async def create_notification(
    notification: Notification,
    notification_service: NotificationService = Depends()
):
    await notification_service.queue_notification(notification)
    return {"id": notification.id, "status": notification.status}

@router.get("/notifications/{notification_id}")
async def get_notification(
    notification_id: str,
    notification_service: NotificationService = Depends()
):
    notification = await notification_service.get_notification(notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    return notification

# services/notification_service.py
import logging
from datetime import datetime, timedelta
import asyncio
from typing import Optional, List
import aioredis
import json
from core.models import Notification, NotificationStatus, NotificationChannel, NotificationPriority
from core.config import settings

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        self.redis = None
        
    async def connect(self):
        if self.redis is None:
            self.redis = await aioredis.create_redis_pool(settings.REDIS_URL)
        
    async def close(self):
        if self.redis is not None:
            self.redis.close()
            await self.redis.wait_closed()
            self.redis = None
    
    def _get_queue_name(self, notification: Notification) -> str:
        """Determine queue name based on channel and priority"""
        priority = notification.priority.value
        channel = notification.channel.value
        return f"notifications:{channel}:{priority}"
    
    async def queue_notification(self, notification: Notification) -> None:
        """Queue a notification for delivery"""
        await self.connect()
        
        # Update status
        notification.status = NotificationStatus.QUEUED
        notification.updated_at = datetime.utcnow()
        
        # Add to queue based on priority
        queue_name = self._get_queue_name(notification)
        await self.redis.lpush(queue_name, notification.json())
        
        # Store for status tracking
        await self.redis.set(
            f"notification:{notification.id}", 
            notification.json(), 
            expire=settings.NOTIFICATION_TTL
        )
        
        logger.info(f"Queued notification {notification.id} to {queue_name}")
    
    async def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Retrieve notification status"""
        await self.connect()
        
        data = await self.redis.get(f"notification:{notification_id}")
        if not data:
            return None
            
        return Notification.parse_raw(data)
    
    async def update_notification_status(
        self, 
        notification_id: str, 
        status: NotificationStatus, 
        error: Optional[str] = None
    ) -> None:
        """Update notification status"""
        await self.connect()
        
        data = await self.redis.get(f"notification:{notification_id}")
        if not data:
            logger.error(f"Notification {notification_id} not found for status update")
            return
            
        notification = Notification.parse_raw(data)
        notification.status = status
        notification.updated_at = datetime.utcnow()
        
        if status == NotificationStatus.DELIVERED:
            notification.sent_at = datetime.utcnow()
        
        if error:
            notification.metadata["last_error"] = error
        
        await self.redis.set(
            f"notification:{notification.id}", 
            notification.json(), 
            expire=settings.NOTIFICATION_TTL
        )

# workers/base_worker.py
from abc import ABC, abstractmethod
import asyncio
import logging
import json
import traceback
from datetime import datetime, timedelta
import aioredis
from core.models import Notification, NotificationStatus, NotificationChannel, NotificationPriority
from services.notification_service import NotificationService

logger = logging.getLogger(__name__)

class BaseWorker(ABC):
    def __init__(
        self,
        channel: NotificationChannel,
        redis_url: str,
        batch_size: int = 10,
        poll_interval: float = 1.0
    ):
        self.channel = channel
        self.redis_url = redis_url
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.redis = None
        self.notification_service = NotificationService()
        self.running = False
        
    async def connect(self):
        if self.redis is None:
            self.redis = await aioredis.create_redis_pool(self.redis_url)
        await self.notification_service.connect()
    
    async def close(self):
        if self.redis is not None:
            self.redis.close()
            await self.redis.wait_closed()
            self.redis = None
        await self.notification_service.close()
        
    async def process_queues(self):
        """Process notifications from queues in priority order"""
        priorities = [
            NotificationPriority.CRITICAL,
            NotificationPriority.HIGH,
            NotificationPriority.NORMAL,
            NotificationPriority.LOW
        ]
        
        for priority in priorities:
            queue_name = f"notifications:{self.channel}:{priority}"
            
            # Get batch of notifications
            notifications_data = await self.redis.lrange(
                queue_name, 0, self.batch_size - 1
            )
            
            if not notifications_data:
                continue
                
            # Process batch
            for notification_json in notifications_data:
                try:
                    notification = Notification.parse_raw(notification_json)
                    await self.process_notification(notification)
                    
                    # Remove from queue
                    await self.redis.lrem(queue_name, 1, notification_json)
                except Exception as e:
                    logger.error(f"Error processing notification: {str(e)}")
                    logger.error(traceback.format_exc())
    
    async def process_notification(self, notification: Notification):
        """Process a single notification"""
        try:
            # Update status
            await self.notification_service.update_notification_status(
                notification.id, NotificationStatus.DELIVERING
            )
            
            # Deliver notification
            await self.deliver(notification)
            
            # Mark as delivered
            await self.notification_service.update_notification_status(
                notification.id, NotificationStatus.DELIVERED
            )
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to deliver notification {notification.id}: {error_message}")
            
            # Handle retries
            notification.retry_count += 1
            if notification.retry_count <= notification.max_retries:
                # Re-queue with exponential backoff
                retry_queue = f"notifications:{self.channel}:retry"
                retry_delay = 60 * (2 ** (notification.retry_count - 1))  # Exponential backoff
                
                await self.redis.zadd(
                    retry_queue, 
                    int(datetime.utcnow().timestamp() + retry_delay),
                    notification.json()
                )
                
                await self.notification_service.update_notification_status(
                    notification.id, 
                    NotificationStatus.PENDING,
                    error=error_message
                )
            else:
                # Mark as failed
                await self.notification_service.update_notification_status(
                    notification.id, 
                    NotificationStatus.FAILED,
                    error=error_message
                )
    
    async def process_retries(self):
        """Process notifications in the retry queue that are due"""
        retry_queue = f"notifications:{self.channel}:retry"
        now = datetime.utcnow().timestamp()
        
        # Get notifications due for retry
        notifications_data = await self.redis.zrangebyscore(
            retry_queue, 0, now
        )
        
        for notification_json in notifications_data:
            notification = Notification.parse_raw(notification_json)
            
            # Re-queue notification
            await self.notification_service.queue_notification(notification)
            
            # Remove from retry queue
            await self.redis.zrem(retry_queue, notification_json)
    
    @abstractmethod
    async def deliver(self, notification: Notification) -> bool:
        """Implement in subclass to deliver notification via specific channel"""
        pass
    
    async def run(self):
        """Main worker loop"""
        await self.connect()
        self.running = True
        
        logger.info(f"Starting {self.channel} notification worker")
        
        while self.running:
            try:
                await self.process_queues()
                await self.process_retries()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in worker: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(self.poll_interval * 2)  # Back off on error
        
        await self.close()
        logger.info(f"Stopped {self.channel} notification worker")
    
    def stop(self):
        """Signal worker to stop"""
        self.running = False

# workers/email_worker.py
from workers.base_worker import BaseWorker
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from core.models import Notification, NotificationChannel
from core.config import settings

class EmailWorker(BaseWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(channel=NotificationChannel.EMAIL, *args, **kwargs)
    
    async def deliver(self, notification: Notification) -> bool:
        """Send email notification"""
        # Create message
        message = MIMEMultipart()
        message["From"] = settings.EMAIL_FROM
        message["To"] = notification.recipient
        message["Subject"] = notification.subject or "Notification"
        
        # Add body
        if isinstance(notification.content, str):
            message.attach(MIMEText(notification.content, "plain"))
        else:
            # Handle HTML content if needed
            if "html" in notification.content:
                message.attach(MIMEText(notification.content["html"], "html"))
            if "text" in notification.content:
                message.attach(MIMEText(notification.content["text"], "plain"))
        
        # Send email
        await aiosmtplib.send(
            message,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USERNAME,
            password=settings.SMTP_PASSWORD,
            use_tls=settings.SMTP_USE_TLS
        )
        
        return True

# main.py
import asyncio
import logging
import signal
from typing import List
import uvicorn
from fastapi import FastAPI
from core.config import settings
from api.router import router
from workers.email_worker import EmailWorker
from workers.sms_worker import SMSWorker
from workers.push_worker import PushWorker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Notification Service")
app.include_router(router, prefix="/api/v1")

# Worker instances
workers: List[BaseWorker] = []
worker_tasks: List[asyncio.Task] = []

@app.on_event("startup")
async def startup_event():
    """Start worker processes on application startup"""
    global workers, worker_tasks
    
    # Create workers
    workers = [
        EmailWorker(redis_url=settings.REDIS_URL),
        SMSWorker(redis_url=settings.REDIS_URL),
        PushWorker(redis_url=settings.REDIS_URL)
    ]
    
    # Start worker tasks
    for worker in workers:
        task = asyncio.create_task(worker.run())
        worker_tasks.append(task)
    
    logger.info("All notification workers started")

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shut down workers"""
    global workers, worker_tasks
    
    logger.info("Shutting down notification workers...")
    
    # Signal workers to stop
    for worker in workers:
        worker.stop()
    
    # Wait for workers to finish (with timeout)
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    logger.info("All notification workers stopped")

def handle_signals():
    """Set up signal handlers for graceful shutdown"""
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: asyncio.create_task(app.shutdown()))

if __name__ == "__main__":
    handle_signals()
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
```

**Key Design Considerations:**

1. **Decoupled Architecture**
    
    - Queue-based approach separates notification processing from API handling
    - Channel-specific workers allow independent scaling based on load
2. **Reliability**
    
    - Persistent queue storage ensures notifications aren't lost if workers crash
    - Intelligent retry mechanism with exponential backoff
    - Tracking of notification status throughout the lifecycle
3. **Scalability**
    
    - Horizontal scaling by adding more workers
    - Priority-based processing ensures important notifications are delivered first
    - Batch processing for efficiency under high load
4. **Resilience**
    
    - Error handling at every stage
    - Circuit breakers could be added to prevent hammering failing services
    - Dead letter queues for notifications that exceed retry limits
5. **Observability**
    
    - Comprehensive logging
    - Status tracking for all notifications
    - Metrics collection could be added with Prometheus

**Additional Production Considerations:**

1. **Distributed Deployment**
    
    - API servers and workers can be deployed separately
    - Kubernetes for orchestration, with separate deployments for API and each worker type
2. **Configuration Management**
    
    - Environment-specific settings for development, staging, and production
    - Secrets management for API keys and credentials
3. **Rate Limiting**
    
    - Implement per-channel and per-recipient rate limits
    - Respect third-party service limits to avoid account suspension
4. **Testing Strategy**
    
    - Unit tests for business logic
    - Integration tests with mocked external services
    - End-to-end tests for critical paths


## System Design & Architecture (continued)

### 4. Question (continued): How would you design a highly scalable notification system...

**Additional Production Considerations (continued):**

5. **System Evolution**
    
    - Template management system for notification content
    - Localization support for international users
    - A/B testing capabilities for notification effectiveness
    - User preference management for notification opt-in/opt-out
6. **Monitoring and Alerting**
    
    - Grafana dashboards for real-time monitoring
    - Alerting on queue depth, error rates, and delivery latency
    - SLO definition and tracking for notification delivery

In a real-world implementation I led, we built a similar system that started with 50K notifications per day and scaled to over 5 million. The key insights gained were:

1. Email delivery was the most prone to delays, so we built a dynamic worker scaling system that increased worker count during peak times
2. Adding proper batching with third-party providers significantly reduced API costs and improved throughput
3. Implementing a channel fallback system (e.g., SMS if push notification fails) increased overall delivery success rates by 12%

### 5. Question: How would you approach building a real-time analytics pipeline for processing and analyzing high-volume log data (5+ GB/hour) using Python?

**Answer:**

Building a real-time analytics pipeline for high-volume log data requires careful consideration of data ingestion, processing, storage, and visualization. Here's how I would approach it:

**Architecture Overview:**

```
Log Sources → Stream Ingestion → Processing → Storage → Query Layer → Visualization
  |               |                |            |           |             |
  |          (Kafka/Kinesis)  (Spark/Flink)  (Various)    (APIs)     (Dashboards)
  |
Instrumentation
```

**1. Data Collection & Ingestion**

I'd implement a distributed streaming platform to handle high-volume data:

```python
# Log producer example with Kafka
from confluent_kafka import Producer
import socket
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class LogEventProducer:
    def __init__(self, bootstrap_servers, topic_name, batch_size=100, flush_interval=1.0):
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'client.id': socket.gethostname(),
            'linger.ms': 50,  # Batch messages
            'compression.type': 'snappy',  # Enable compression
            'queue.buffering.max.messages': 100000,
            'queue.buffering.max.kbytes': 50000,
            'batch.num.messages': batch_size
        })
        self.topic = topic_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.message_count = 0
    
    def delivery_callback(self, err, msg):
        """Called once for each message produced to indicate delivery result."""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            self.message_count += 1
            # Auto-flush based on interval
            if time.time() - self.last_flush > self.flush_interval:
                self.producer.flush()
                self.last_flush = time.time()
    
    def send_log_event(self, log_event):
        """Send a log event to Kafka."""
        # Add timestamp if not present
        if 'timestamp' not in log_event:
            log_event['timestamp'] = datetime.utcnow().isoformat()
        
        # Serialize to JSON
        try:
            log_json = json.dumps(log_event).encode('utf-8')
            
            # Send to Kafka
            self.producer.produce(
                self.topic,
                key=str(log_event.get('service_id', '')).encode('utf-8'),
                value=log_json,
                callback=self.delivery_callback
            )
        except Exception as e:
            logger.error(f"Error sending log event: {str(e)}")
    
    def flush(self):
        """Flush any pending messages."""
        self.producer.flush()
        self.last_flush = time.time()

# Usage
producer = LogEventProducer(
    bootstrap_servers='kafka1:9092,kafka2:9092,kafka3:9092',
    topic_name='application_logs'
)

# Example log event
log_event = {
    'service_id': 'user-service',
    'level': 'ERROR',
    'message': 'Database connection failed',
    'context': {
        'user_id': '12345',
        'request_id': 'abc-123',
        'endpoint': '/api/users'
    },
    'host': 'app-server-42',
    'environment': 'production'
}

producer.send_log_event(log_event)
producer.flush()
```

**2. Stream Processing**

I'd use Apache Spark Streaming for real-time processing:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, count, expr
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, MapType

# Define schema for log events
log_schema = StructType([
    StructField("timestamp", TimestampType()),
    StructField("service_id", StringType()),
    StructField("level", StringType()),
    StructField("message", StringType()),
    StructField("context", MapType(StringType(), StringType())),
    StructField("host", StringType()),
    StructField("environment", StringType())
])

# Initialize Spark session
spark = SparkSession.builder \
    .appName("LogAnalyticsPipeline") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.default.parallelism", "16") \
    .getOrCreate()

# Read from Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka1:9092,kafka2:9092,kafka3:9092") \
    .option("subscribe", "application_logs") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .option("maxOffsetsPerTrigger", "10000") \
    .load()

# Parse JSON data
parsed_df = kafka_df \
    .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
    .select(
        from_json(col("value"), log_schema).alias("data")
    ) \
    .select("data.*")

# Real-time analytics
# 1. Error rate by service (5-minute sliding window)
error_rate = parsed_df \
    .filter(col("level") == "ERROR") \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        col("service_id"),
        window(col("timestamp"), "5 minutes", "1 minute")
    ) \
    .count() \
    .withColumnRenamed("count", "error_count")

# Write to Postgres for real-time dashboards
error_rate_query = error_rate \
    .writeStream \
    .outputMode("append") \
    .foreachBatch(lambda df, epoch_id: df.write
        .format("jdbc")
        .option("url", "jdbc:postgresql://db:5432/analytics")
        .option("dbtable", "error_rates")
        .option("user", "analytics_user")
        .option("password", "****")
        .mode("append")
        .save()
    ) \
    .trigger(processingTime="1 minute") \
    .start()

# 2. Log throughput by service and host
throughput = parsed_df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        col("service_id"),
        col("host"),
        window(col("timestamp"), "1 minute")
    ) \
    .count() \
    .withColumnRenamed("count", "log_count")

# 3. Alert on error spikes
error_alerts = parsed_df \
    .filter(col("level") == "ERROR") \
    .withWatermark("timestamp", "5 minutes") \
    .groupBy(
        col("service_id"),
        window(col("timestamp"), "1 minute")
    ) \
    .count() \
    .filter(col("count") > 100)  # Alert threshold

# Write to alert topic
alert_query = error_alerts \
    .writeStream \
    .outputMode("update") \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka1:9092,kafka2:9092") \
    .option("topic", "error_alerts") \
    .option("checkpointLocation", "/tmp/spark/checkpoints/alerts") \
    .start()

# 4. Store all logs for long-term storage
storage_query = parsed_df \
    .writeStream \
    .format("parquet") \
    .option("path", "s3a://logs-analytics/processed-logs") \
    .option("checkpointLocation", "/tmp/spark/checkpoints/storage") \
    .partitionBy("service_id", "level", "year", "month", "day") \
    .trigger(processingTime="5 minutes") \
    .start()

# Wait for all queries to terminate
spark.streams.awaitAnyTermination()
```

**3. Data Storage Strategy**

For a comprehensive solution, I'd implement a multi-tier storage strategy:

```python
# Hot tier: Time-series database for recent data (InfluxDB example)
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

# Initialize InfluxDB client
influx_client = InfluxDBClient(
    url="http://influxdb:8086",
    token="YOUR_API_TOKEN",
    org="analytics-org"
)

# Configure write API with batching
write_api = influx_client.write_api(
    write_options=WriteOptions(
        batch_size=5000,
        flush_interval=10_000,
        jitter_interval=2_000,
        retry_interval=5_000,
        max_retries=5,
        max_retry_delay=30_000,
        exponential_base=2
    )
)

def write_logs_to_influx(processed_logs_df):
    """Write processed logs to InfluxDB."""
    for log in processed_logs_df.collect():
        # Create a data point
        point = Point("application_logs") \
            .tag("service_id", log.service_id) \
            .tag("level", log.level) \
            .tag("host", log.host) \
            .tag("environment", log.environment) \
            .field("message", log.message)
        
        # Add context fields
        if log.context:
            for key, value in log.context.items():
                point = point.tag(f"ctx_{key}", value)
        
        # Write with timestamp
        point.time(log.timestamp)
        write_api.write(bucket="logs", record=point)

# For cold storage, we'll use Parquet files on S3 (already shown in Spark example)
# This provides cost-effective long-term storage with good query performance
```

**4. Query and Analytics API**

I'd build a FastAPI service for analytics queries:

```python
from fastapi import FastAPI, Depends, HTTPException, Query
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import time
from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi
import pandas as pd
import asyncio
from functools import lru_cache

app = FastAPI(title="Log Analytics API")

# Configuration
INFLUXDB_URL = "http://influxdb:8086"
INFLUXDB_TOKEN = "YOUR_API_TOKEN"
INFLUXDB_ORG = "analytics-org"
BUCKET = "logs"

# Cache frequently accessed queries
@lru_cache(maxsize=100)
def get_influx_client() -> InfluxDBClient:
    return InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

# Dependency for query API
def get_query_api() -> QueryApi:
    client = get_influx_client()
    return client.query_api()

@app.get("/api/v1/logs/errorRate")
async def get_error_rate(
    service_id: Optional[str] = None,
    start_time: datetime = Query(default_factory=lambda: datetime.utcnow() - timedelta(hours=1)),
    end_time: datetime = Query(default_factory=lambda: datetime.utcnow()),
    interval: str = "5m",
    query_api: QueryApi = Depends(get_query_api)
):
    """Get error rate over time for specified service."""
    # Build Flux query
    service_filter = f'r.service_id == "{service_id}"' if service_id else 'true'
    query = f'''
    from(bucket: "{BUCKET}")
        |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
        |> filter(fn: (r) => r._measurement == "application_logs")
        |> filter(fn: (r) => {service_filter})
        |> filter(fn: (r) => r.level == "ERROR")
        |> aggregateWindow(every: {interval}, fn: count)
        |> yield(name: "error_count")
    '''
    
    # Execute query
    result = query_api.query_data_frame(query)
    
    # Process and format results
    if result.empty:
        return {"data": []}
    
    # Transform to time series format
    result = result.rename(columns={"_time": "timestamp", "_value": "count"})
    data = result[["timestamp", "count", "service_id"]].to_dict(orient="records")
    
    return {"data": data}

@app.get("/api/v1/logs/search")
async def search_logs(
    query: str,
    service_id: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = 100,
    start_time: datetime = Query(default_factory=lambda: datetime.utcnow() - timedelta(hours=24)),
    end_time: datetime = Query(default_factory=lambda: datetime.utcnow()),
    query_api: QueryApi = Depends(get_query_api)
):
    """Search logs with various filters."""
    # Build filters
    filters = []
    if service_id:
        filters.append(f'r.service_id == "{service_id}"')
    if level:
        filters.append(f'r.level == "{level}"')
    if query:
        filters.append(f'r.message =~ /{query}/')
    
    filter_expr = " and ".join(filters) if filters else "true"
    
    # Build Flux query
    flux_query = f'''
    from(bucket: "{BUCKET}")
        |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
        |> filter(fn: (r) => r._measurement == "application_logs")
        |> filter(fn: (r) => {filter_expr})
        |> limit(n: {limit})
    '''
    
    # Execute query
    result = query_api.query_data_frame(flux_query)
    
    # Process results
    if result.empty:
        return {"logs": []}
    
    # Format response
    logs = []
    for _, row in result.iterrows():
        log = {
            "timestamp": row.get("_time").isoformat(),
            "service_id": row.get("service_id"),
            "level": row.get("level"),
            "message": row.get("message"),
            "host": row.get("host")
        }
        logs.append(log)
    
    return {"logs": logs}

# More endpoints for advanced analytics...
```

**5. System Optimizations for High Volume**

For handling 5+ GB/hour (approximately 100+ million log events daily), I would implement:

1. **Partitioning Strategy**
    
    ```python
    # Smart partitioning in Spark
    def calculate_optimal_partitions(df):
        """Dynamically calculate optimal partitions based on data size."""
        # Get estimated size in bytes
        estimated_size = df.select("*").explain("cost")
        # Calculate partitions (approximately 128MB per partition)
        return max(8, int(estimated_size / (128 * 1024 * 1024)))
    
    # In Spark job
    optimal_partitions = calculate_optimal_partitions(parsed_df)
    repartitioned_df = parsed_df.repartition(optimal_partitions)
    ```
    
2. **Intelligent Sampling for High-Cardinality Data**
    
    ```python
    from pyspark.sql.functions import rand
    
    # Sample logs for high-volume analysis when precision isn't critical
    def sample_logs_for_analysis(df, sampling_rate=0.1):
        """Create a statistically significant sample for expensive analytics."""
        return df.filter(rand() < sampling_rate)
        
    # For tracking rare events, use stratified sampling to ensure representation
    def stratified_sample(df):
        # Keep all ERROR logs, sample others
        errors = df.filter(col("level") == "ERROR")
        others = df.filter(col("level") != "ERROR").sample(0.1)
        return errors.union(others)
    ```
    
3. **Data Expiration and Tiering**
    
    ```python
    # In production code, implement data lifecycle policies
    
    # Hot tier: Last 7 days in InfluxDB (full resolution)
    # Warm tier: Last 90 days in object storage (hourly aggregations)
    # Cold tier: 1+ year in object storage (daily aggregations)
    
    # Example: Create downsampled aggregations for historical data
    def create_downsampled_data(year, month):
        """Create hourly and daily summaries for historical data."""
        spark = get_spark_session()
        
        # Read raw logs
        logs_path = f"s3a://logs-analytics/processed-logs/year={year}/month={month}/"
        raw_logs = spark.read.parquet(logs_path)
        
        # Hourly aggregations
        hourly_metrics = raw_logs.groupBy(
            "service_id", "level",
            year("timestamp").alias("year"),
            month("timestamp").alias("month"),
            dayofmonth("timestamp").alias("day"),
            hour("timestamp").alias("hour")
        ).agg(
            count("*").alias("event_count"),
            # More aggregations...
        )
        
        # Daily aggregations
        daily_metrics = hourly_metrics.groupBy(
            "service_id", "level", "year", "month", "day"
        ).agg(
            sum("event_count").alias("event_count"),
            # More aggregations...
        )
        
        # Write to appropriate storage
        hourly_metrics.write.parquet(
            f"s3a://logs-analytics/hourly-metrics/year={year}/month={month}/"
        )
        
        daily_metrics.write.parquet(
            f"s3a://logs-analytics/daily-metrics/year={year}/month={month}/"
        )
    ```
    

**Key Design Considerations for Production:**

1. **Optimized Infrastructure**
    
    - Kafka cluster sized appropriately (5-10 brokers with 32GB RAM each)
    - Spark cluster with auto-scaling capabilities (Databricks or EMR)
    - Isolated resources for critical components
    - Event-driven architecture with clear separation of concerns
2. **Data Locality and Distribution**
    
    - Co-locate processing with storage when possible
    - Distribute data processing across multiple regions for global applications
    - Replicate critical analytical outputs for high availability
3. **Operational Excellence**
    
    - Circuit breakers to prevent cascading failures
    - Backpressure mechanisms throughout the pipeline
    - End-to-end monitoring with alerting
    - Dead-letter queues for logs that fail processing
4. **Cost Optimization**
    
    - Data lifecycle management with automatic tiering
    - Spot instances for non-critical processing
    - Intelligent partitioning for efficient storage and queries
    - Reserved capacity for predictable workloads
5. **Security and Compliance**
    
    - PII detection and masking in real-time
    - Audit trail for all data access
    - Encryption at rest and in transit
    - Role-based access controls for analytical APIs

In a real-world implementation at a previous company, this architecture successfully processed over 10 billion log events daily (~20TB) while maintaining sub-second query performance for recent data and providing cost-effective storage for historical analysis.

## Performance and Optimization

### 6. Question: You've been tasked with optimizing a slow-performing Python application. Walk through your approach to diagnosing and addressing performance bottlenecks.

**Answer:**

Performance optimization is a methodical process that requires both investigation and targeted improvements. Here's my comprehensive approach based on real production experience:

**1. Establish Baselines and Metrics**

First, I establish quantifiable metrics to measure improvements:

```python
import time
import statistics
from memory_profiler import memory_usage
import cProfile
import pstats
from functools import wraps

def performance_timer(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        times = []
        for _ in range(5):  # Run multiple times for statistical significance
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        print(f"Function {func.__name__}:")
        print(f"  Avg time: {statistics.mean(times):.6f} seconds")
        print(f"  Min time: {min(times):.6f} seconds")
        print(f"  Max time: {max(times):.6f} seconds")
        print(f"  Std dev:  {statistics.stdev(times):.6f} seconds")
        return result
    return wrapper

def memory_profile(func):
    """Decorator to measure memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        mem_usage, result = memory_usage(
            (func, args, kwargs),
            retval=True,
            interval=0.1,
            timeout=None
        )
        print(f"Function {func.__name__}:")
        print(f"  Peak memory usage: {max(mem_usage) - min(mem_usage):.2f} MiB")
        return result
    return wrapper

def profile_func(func):
    """Profile function execution using cProfile."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            stats = pstats.Stats(profile)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Print top 20 functions by cumulative time
    return wrapper

# Example usage on our slow function
@performance_timer
@memory_profile
@profile_func
def slow_function(data):
    # Function to optimize
    result = process_data(data)
    return result
```

**2. Identify Performance Bottlenecks**

I use a combination of profiling tools to identify bottlenecks:

```python
# Targeted profiling with line_profiler
# pip install line_profiler
%load_ext line_profiler

%lprun -f problematic_function problematic_function(sample_data)

# Sample output might show slow lines like:
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#     30        1       5000.0   5000.0     95.0      for item in large_list:
#     31     1000       5000.0      5.0     95.0          do_expensive_operation(item)

# Visualize call graph with snakeviz
# pip install snakeviz
import cProfile
cProfile.run('slow_function(sample_data)', 'profile_stats')

# In terminal:
# $ snakeviz profile_stats

# For long-running applications, use py-spy for sampling without modifying code
# pip install py-spy
# $ py-spy record -o profile.svg --pid 12345
```

**3. Common Bottlenecks and Optimizations**

Based on profiling, I focus on specific optimization strategies:

**A. Inefficient Algorithms and Data Structures**

```python
# Before: O(n²) nested loop approach
def find_pairs_slow(numbers, target_sum):
    pairs = []
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if numbers[i] + numbers[j] == target_sum:
                pairs.append((numbers[i], numbers[j]))
    return pairs

# After: O(n) approach with hash map
def find_pairs_fast(numbers, target_sum):
    pairs = []
    seen = set()
    for num in numbers:
        complement = target_sum - num
        if complement in seen:
            pairs.append((complement, num))
        seen.add(num)
    return pairs

# Before: Inefficient data access pattern
def process_repeated_lookups(data, keys):
    result = []
    for key in keys:  # keys may contain repeated values
        if key in data:  # O(n) for lists, O(1) for dict but still repeated
            result.append(data[key])
    return result

# After: Preprocess for efficient access
def process_repeated_lookups_optimized(data, keys):
    # Convert data to dict if it's not already
    data_dict = data if isinstance(data, dict) else {item['key']: item for item in data}
    # Process all keys at once
    return [data_dict.get(key) for key in keys]
```

**B. Excessive I/O Operations**

```python
# Before: Reading a file line by line with multiple operations
def process_log_file_slow(filename):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            if 'ERROR' in line:
                # Parse line
                parts = line.strip().split(',')
                timestamp = parts[0]
                message = parts[1]
                # Do some processing
                processed = process_message(message)
                results.append((timestamp, processed))
    return results

# After: Batch processing with optimized I/O
def process_log_file_fast(filename):
    # Read in larger chunks
    chunk_size = 8192  # 8KB chunks
    results = []
    
    with open(filename, 'r', buffering=chunk_size) as f:
        # Process file in chunks
        buffer = ""
        
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
                
            buffer += chunk
            lines = buffer.split('\n')
            
            # Keep the last partial line for the next iteration
            buffer = lines.pop()
            
            # Process complete lines
            error_lines = [l for l in lines if 'ERROR' in l]
            
            # Batch process the error lines
            if error_lines:
                batch_results = batch_process_lines(error_lines)
                results.extend(batch_results)
    
    # Process any remaining content
    if buffer:
        if 'ERROR' in buffer:
            processed = process_line(buffer)
            results.append(processed)
    
    return results

def batch_process_lines(lines):
    """Process multiple lines at once for efficiency."""
    # Extract all fields at once
    parsed = [line.strip().split(',') for line in lines]
    timestamps = [p[0] for p in parsed]
    messages = [p[1] for p in parsed]
    
    # Process all messages in one batch
    processed_messages = batch_process_messages(messages)
    
    # Combine results
    return list(zip(timestamps, processed_messages))
```

**C. Database Optimizations**

```python
import sqlalchemy as sa
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

# Before: N+1 query problem
def get_users_with_orders_slow(session):
    users = session.query(User).all()
    
    # For each user, make a separate query to get their orders
    for user in users:
        user.orders = session.query(Order).filter(Order.user_id == user.id).all()
    
    return users

# After: Optimized with join and eager loading
def get_users_with_orders_fast(session):
    return session.query(User).options(
        joinedload(User.orders)
    ).all()

# Before: Inefficient filtering with Python
def find_active_premium_users_slow(session):
    # Get all users first, then filter in Python
    all_users = session.query(User).all()
    return [user for user in all_users 
            if user.is_active and user.subscription_type == 'premium']

# After: Let the database do the filtering
def find_active_premium_users_fast(session):
    return session.query(User).filter(
        User.is_active == True,
        User.subscription_type == 'premium'
    ).all()

# Before: Loading unnecessary data
def get_user_stats_slow(session, user_id):
    # Load the entire user object with all columns
    user = session.query(User).filter(User.id == user_id).one()
    return {
        'name': user.name,
        'post_count': user.posts_count,
        'comment_count': user.comments_count
    }

# After: Load only what's needed
def get_user_stats_fast(session, user_id):
    # Select only the required columns
    result = session.query(
        User.name, 
        User.posts_count, 
        User.comments_count
    ).filter(User.id == user_id).one()
    
    return {
        'name': result.name,
        'post_count': result.posts_count,
        'comment_count': result.comments_count
    }

# For complex queries, use raw SQL when appropriate
def complex_analytics_query(session, start_date, end_date):
    query = text("""
        SELECT 
            u.id, 
            u.name,
            COUNT(o.id) as order_count,
            SUM(o.total_amount) as total_spent
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.created_at BETWEEN :start_date AND :end_date
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 5
        ORDER BY total_spent DESC
        LIMIT 100
    """)
    
    return session.execute(
        query, 
        {"start_date": start_date, "end_date": end_date}
    ).fetchall()
```

**D. Memory Optimizations**

```python
# Before: Creating large intermediate lists
def process_large_dataset_slow(data):
    # Filter
    filtered = [item for item in data if filter_condition(item)]
    # Transform
    transformed = [transform_item(item) for item in filtered]
    # Group
    grouped = {}
    for item in transformed:
        key = item['category']
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item)
    # Aggregate
    result = {k: aggregate_items(v) for k, v in grouped.items()}
    return result

# After: Using generators to avoid intermediate storage
def process_large_dataset_fast(data):
    # Process in a streaming fashion
    def filtered_items():
        for item in data:
            if filter_condition(item):
                yield item
    
    def transformed_items():
        for item in filtered_items():
            yield transform_
```
## Performance and Optimization (continued)

```python
# Memory Optimizations (continued)
def process_large_dataset_fast(data):
    # Process in a streaming fashion
    def filtered_items():
        for item in data:
            if filter_condition(item):
                yield item
    
    def transformed_items():
        for item in filtered_items():
            yield transform_item(item)
    
    # Group without storing everything
    grouped = {}
    for item in transformed_items():
        key = item['category']
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item)
    
    # Aggregate
    return {k: aggregate_items(v) for k, v in grouped.items()}

# Using generators efficiently
def process_large_file_with_generators(filename):
    def parse_lines():
        with open(filename, 'r') as f:
            for line in f:
                yield parse_line(line)
    
    def filter_valid_records():
        for record in parse_lines():
            if is_valid_record(record):
                yield record
    
    def transform_records():
        for record in filter_valid_records():
            yield transform_record(record)
    
    # Process everything in a memory-efficient streaming fashion
    result = {}
    for record in transform_records():
        category = record['category']
        result[category] = result.get(category, 0) + record['value']
    
    return result

# Use itertools for memory-efficient operations
import itertools
from collections import defaultdict

def process_with_itertools(data_source):
    # Read and parse data efficiently
    records = map(parse_record, data_source)
    
    # Filter records
    valid_records = filter(is_valid, records)
    
    # Group by category with minimal memory usage
    sorted_records = sorted(valid_records, key=lambda x: x['category'])
    grouped = {
        category: list(items)
        for category, items in itertools.groupby(sorted_records, key=lambda x: x['category'])
    }
    
    # Process each group
    return {
        category: process_group(items)
        for category, items in grouped.items()
    }

# For very large datasets, consider using NumPy for efficient memory usage
import numpy as np

def process_numeric_data_efficiently(raw_data):
    # Convert to NumPy array for memory efficiency
    data = np.array(raw_data, dtype=np.float32)  # Use appropriate data type
    
    # NumPy operations are much more memory-efficient than Python loops
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized = (data - mean) / std_dev
    
    # Apply threshold filter
    filtered = normalized[np.abs(normalized) < 3]  # Remove outliers
    
    return {
        'mean': mean.tolist(),
        'std_dev': std_dev.tolist(),
        'filtered_count': filtered.shape[0],
        'filtered_mean': np.mean(filtered, axis=0).tolist()
    }
```

**E. Multi-processing and Concurrency**

```python
import concurrent.futures
import multiprocessing
import asyncio
import aiohttp

# Parallelize CPU-bound tasks
def process_data_parallel(data_chunks):
    """Process multiple chunks of data in parallel using multiprocessing."""
    cpu_count = multiprocessing.cpu_count()
    chunk_size = max(1, len(data_chunks) // cpu_count)
    
    # Split data for parallel processing
    chunks = [data_chunks[i:i+chunk_size] for i in range(0, len(data_chunks), chunk_size)]
    
    # Process in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Combine results
    return combine_results(results)

def process_chunk(chunk):
    """Process a single chunk of data."""
    results = []
    for item in chunk:
        # CPU-intensive processing
        processed = complex_calculation(item)
        results.append(processed)
    return results

# Thread-based concurrency for I/O-bound tasks
def download_all_urls(urls):
    """Download multiple URLs concurrently using threads."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_url = {executor.submit(download_url, url): url for url in urls}
        results = {}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as e:
                results[url] = f"Error: {str(e)}"
    
    return results

def download_url(url):
    """Download a single URL."""
    import requests
    response = requests.get(url, timeout=30)
    return response.text

# Async I/O for highly concurrent operations
async def fetch_all_apis_async(api_urls):
    """Fetch multiple API endpoints concurrently using asyncio."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_json(session, url) for url in api_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map urls to results
        return {url: result for url, result in zip(api_urls, results)}

async def fetch_json(session, url):
    """Fetch a single JSON endpoint."""
    async with session.get(url) as response:
        return await response.json()

# Handle mixed workloads efficiently
def process_mixed_workload(items):
    """Process a workload with both CPU and I/O bound tasks."""
    # Split tasks by type
    io_tasks = [item for item in items if item['type'] == 'io']
    cpu_tasks = [item for item in items if item['type'] == 'cpu']
    
    # Process I/O tasks with threads or asyncio
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        io_futures = {executor.submit(process_io_task, task): task for task in io_tasks}
        io_results = {}
        
        for future in concurrent.futures.as_completed(io_futures):
            task = io_futures[future]
            io_results[task['id']] = future.result()
    
    # Process CPU tasks with processes
    cpu_count = multiprocessing.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        cpu_futures = {executor.submit(process_cpu_task, task): task for task in cpu_tasks}
        cpu_results = {}
        
        for future in concurrent.futures.as_completed(cpu_futures):
            task = cpu_futures[future]
            cpu_results[task['id']] = future.result()
    
    # Combine results
    results = {**io_results, **cpu_results}
    return results
```

**F. Caching and Memoization**

```python
import functools
import time
import hashlib
import pickle
import redis

# Simple function-level memoization
@functools.lru_cache(maxsize=128)
def expensive_calculation(n):
    """Perform an expensive calculation with caching for repeated calls."""
    print(f"Computing for {n} (expensive!)")
    time.sleep(1)  # Simulate expensive operation
    return n * n

# Advanced memoization with custom key and timeout
def advanced_memoize(timeout=3600, maxsize=128):
    """Decorator for advanced memoization with ttl and custom key function."""
    cache = {}
    queue = []  # Simple LRU queue
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Check cache
            if key in cache:
                timestamp, value = cache[key]
                if time.time() - timestamp < timeout:
                    # Update position in LRU queue
                    queue.remove(key)
                    queue.append(key)
                    return value
            
            # Calculate new value
            result = func(*args, **kwargs)
            
            # Update cache
            cache[key] = (time.time(), result)
            queue.append(key)
            
            # Enforce maxsize
            if len(queue) > maxsize:
                oldest_key = queue.pop(0)
                if oldest_key in cache:
                    del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

# Example using the advanced memoize
@advanced_memoize(timeout=60, maxsize=256)
def fetch_user_data(user_id):
    """Fetch user data from a database."""
    print(f"Fetching data for user {user_id}")
    # Simulate database lookup
    time.sleep(2)
    return {"user_id": user_id, "name": f"User {user_id}", "age": user_id * 2}

# Distributed caching with Redis for multi-process/multi-server applications
class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0, default_ttl=3600):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.default_ttl = default_ttl
    
    def cached(self, ttl=None):
        """Decorator for Redis-based caching."""
        ttl = ttl or self.default_ttl
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
                cache_key = f"cache:{hashlib.md5(':'.join(key_parts).encode()).hexdigest()}"
                
                # Try to get from cache
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pickle.loads(cached_data)
                
                # Calculate result
                result = func(*args, **kwargs)
                
                # Store in cache
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    pickle.dumps(result)
                )
                
                return result
            return wrapper
        return decorator

# Using Redis cache
redis_cache = RedisCache(host='redis-server', port=6379)

@redis_cache.cached(ttl=300)  # Cache for 5 minutes
def get_product_recommendations(user_id, product_id):
    """Get personalized product recommendations."""
    print(f"Calculating recommendations for user {user_id}, product {product_id}")
    # Simulate recommendation engine
    time.sleep(3)
    return [
        {"id": 101, "score": 0.92},
        {"id": 143, "score": 0.85},
        {"id": 218, "score": 0.78}
    ]
```

**4. Compile Critical Sections (Cython)**

For performance-critical sections, I might use Cython:

```python
# slow_math.py - Pure Python version
def slow_vector_dot(a, b):
    """Compute the dot product of two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# fast_math.pyx - Cython version
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# Type declarations for performance
def fast_vector_dot(np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
    """Compute the dot product of two vectors (Cython version)."""
    if a.shape[0] != b.shape[0]:
        raise ValueError("Vectors must have the same length")
        
    cdef int i
    cdef int n = a.shape[0]
    cdef double result = 0.0
    
    for i in range(n):
        result += a[i] * b[i]
        
    return result
```

**5. Use Appropriate Libraries**

```python
# Before: Manual CSV processing
def process_csv_slow(filename):
    result = []
    with open(filename, 'r') as f:
        # Skip header
        header = next(f).strip().split(',')
        
        for line in f:
            values = line.strip().split(',')
            row = {header[i]: values[i] for i in range(len(header))}
            
            # Process numeric values
            if row['age'].isdigit():
                row['age'] = int(row['age'])
            else:
                row['age'] = 0
                
            if row['income']:
                row['income'] = float(row['income'])
            else:
                row['income'] = 0.0
                
            result.append(row)
            
    return result

# After: Using specialized libraries
import pandas as pd
import numpy as np

def process_csv_fast(filename):
    # Let pandas handle parsing with proper types
    df = pd.read_csv(
        filename,
        dtype={
            'name': str,
            'age': 'Int64',  # Handles missing values better than int
            'income': 'float64'
        }
    )
    
    # Clean data
    df['age'] = df['age'].fillna(0)
    df['income'] = df['income'].fillna(0.0)
    
    # If you need dict format
    return df.to_dict('records')

# Before: Calculating statistics manually
def calculate_stats_slow(data):
    n = len(data)
    total = sum(data)
    mean = total / n
    
    squared_diff_sum = sum((x - mean) ** 2 for x in data)
    variance = squared_diff_sum / n
    std_dev = variance ** 0.5
    
    sorted_data = sorted(data)
    median = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    
    return {
        'mean': mean,
        'median': median,
        'std_dev': std_dev,
        'min': min(data),
        'max': max(data)
    }

# After: Using NumPy for vectorized calculations
import numpy as np

def calculate_stats_fast(data):
    # Convert to numpy array once
    data_array = np.array(data)
    
    return {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std_dev': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array)
    }
```

**6. Real-World Example: Optimizing a Data Processing Pipeline**

Here's a real case I optimized:

```python
# Original slow version of a data processing pipeline
def process_sales_data_slow(sales_file, product_file):
    # Load sales data
    with open(sales_file, 'r') as f:
        sales = [line.strip().split(',') for line in f.readlines()[1:]]
    
    # Load product data
    with open(product_file, 'r') as f:
        products = [line.strip().split(',') for line in f.readlines()[1:]]
    
    # Create product lookup dict
    product_dict = {}
    for p in products:
        product_dict[p[0]] = {
            'name': p[1],
            'category': p[2],
            'price': float(p[3])
        }
    
    # Process sales
    results = []
    for sale in sales:
        sale_id = sale[0]
        date = sale[1]
        customer_id = sale[2]
        product_id = sale[3]
        quantity = int(sale[4])
        
        if product_id in product_dict:
            product = product_dict[product_id]
            
            # Calculate sale amount
            amount = quantity * product['price']
            
            # Append to results
            results.append({
                'sale_id': sale_id,
                'date': date,
                'customer_id': customer_id,
                'product_id': product_id,
                'product_name': product['name'],
                'category': product['category'],
                'quantity': quantity,
                'price': product['price'],
                'amount': amount
            })
    
    # Calculate summary by category
    category_sales = {}
    for result in results:
        category = result['category']
        amount = result['amount']
        
        if category not in category_sales:
            category_sales[category] = 0
        
        category_sales[category] += amount
    
    return {
        'sales': results,
        'category_summary': category_sales
    }

# Optimized version
import pandas as pd
import numpy as np
from datetime import datetime

def process_sales_data_fast(sales_file, product_file):
    # Use pandas for efficient CSV parsing
    sales_df = pd.read_csv(sales_file)
    products_df = pd.read_csv(product_file)
    
    # Optimize column types for memory usage
    sales_df['quantity'] = sales_df['quantity'].astype('int32')
    products_df['price'] = products_df['price'].astype('float32')
    
    # Efficient join operation
    results_df = sales_df.merge(
        products_df,
        on='product_id',
        how='inner'
    )
    
    # Vectorized calculation
    results_df['amount'] = results_df['quantity'] * results_df['price']
    
    # Calculate category summary efficiently
    category_summary = results_df.groupby('category')['amount'].sum().to_dict()
    
    # Convert to dict format if needed
    results = results_df.to_dict('records')
    
    return {
        'sales': results,
        'category_summary': category_summary
    }

# Further optimization for very large files
def process_sales_data_chunked(sales_file, product_file, chunk_size=10000):
    # Load products (usually smaller) into memory
    products_df = pd.read_csv(product_file)
    
    # Process sales in chunks
    category_summary = {}
    all_results = []
    
    # Create chunked iterator
    sales_chunks = pd.read_csv(sales_file, chunksize=chunk_size)
    
    for chunk in sales_chunks:
        # Process this chunk
        chunk_result = chunk.merge(
            products_df,
            on='product_id',
            how='inner'
        )
        
        # Calculate amounts
        chunk_result['amount'] = chunk_result['quantity'] * chunk_result['price']
        
        # Update category summary
        chunk_summary = chunk_result.groupby('category')['amount'].sum().to_dict()
        for category, amount in chunk_summary.items():
            category_summary[category] = category_summary.get(category, 0) + amount
        
        # Store results
        all_results.append(chunk_result)
    
    # Combine results if needed
    if all_results:
        results_df = pd.concat(all_results)
        results = results_df.to_dict('records')
    else:
        results = []
    
    return {
        'sales': results,
        'category_summary': category_summary
    }
```

**Summary of My Optimization Methodology:**

1. **Measure First, Optimize Later**
    
    - Establish clear performance metrics (timing, memory, database calls)
    - Use profiling to identify actual bottlenecks
    - Focus on the 20% of code that causes 80% of performance issues
2. **Address Core Algorithmic Issues First**
    
    - Improve algorithmic complexity where possible (O(n²) → O(n))
    - Choose appropriate data structures for specific operations
    - Minimize unnecessary work (early termination, smart filtering)
3. **Database and I/O Optimization**
    
    - Batch operations when possible
    - Leverage database capabilities rather than processing in Python
    - Use connection pooling, prepared statements, and query optimization
4. **Concurrency and Parallelism**
    
    - Choose the right tool: threading for I/O, multiprocessing for CPU
    - Minimize thread synchronization overhead
    - Consider async for highly concurrent I/O operations
5. **Memory Management**
    
    - Use generators for large datasets
    - Process data in chunks
    - Leverage NumPy for numeric work
    - Control object creation and lifecycle
6. **Targeted Caching**
    
    - Apply caching at appropriate levels (function, request, application)
    - Use distributed caching for multi-server applications
    - Implement proper cache invalidation strategies
7. **Use Specialized Libraries**
    
    - Pandas for data manipulation
    - NumPy for numerical computations
    - Use C extensions or Cython for critical sections

This methodology has helped me achieve 10-100x performance improvements in several real-world applications, most notably:

- A data processing pipeline that went from taking 45 minutes to under 2 minutes
- A web scraping system that scaled from 10K to 1M pages per day
- An API service that reduced p99 latency from 2.5 seconds to 150ms

The key is always to start with measurement, focus on the biggest bottlenecks, and iterate with continuous testing to ensure improvements are real and don't introduce regressions.

## Data Engineering & ETL

### 7. Question: Describe how you would design and implement a data pipeline that processes millions of records daily from various sources (CSV files, APIs, and databases) for analytics. Include error handling, monitoring, and scaling considerations.

**Answer:**

Based on my experience building high-volume data pipelines in production, I'll outline a comprehensive approach for processing millions of records daily from diverse sources.

**Architecture Overview:**

```
Sources → Extraction → Transformation → Loading → Analytics
  ↑               ↓           ↓           ↓          ↓
  └───────── Orchestration & Monitoring ─────────────┘
```

**1. Data Extraction Layer**

I'd design flexible connectors for each source type:

```python
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any, Iterator, Optional
import pandas as pd
import requests
import psycopg2
import pymysql
import sqlalchemy
from sqlalchemy import create_engine
import boto3
import json
import csv
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for all data sources."""
    
    def __init__(self, source_config: Dict[str, Any]):
        self.source_config = source_config
        self.name = source_config.get('name', 'unnamed_source')
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """Validate that the source configuration is correct."""
        pass
    
    @abstractmethod
    def extract_data(self, context: Dict[str, Any] = None) -> Iterator[Dict[str, Any]]:
        """Extract data from the source, yielding batches of records."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """Return the schema of the data source."""
        pass

# File-based sources
class CSVSource(DataSource):
    """Extracts data from CSV files."""
    
    def validate_config(self) -> None:
        required_fields = ['file_path']
        missing = [f for f in required_fields if f not in self.source_config]
        if missing:
            raise ValueError(f"Missing required fields for CSVSource: {missing}")
    
    def extract_data(self, context: Dict[str, Any] = None) -> Iterator[Dict[str, Any]]:
        file_path = self.source_config['file_path']
        chunk_size = self.source_config.get('chunk_size', 10000)
        delimiter = self.source_config.get('delimiter', ',')
        encoding = self.source_config.get('encoding', 'utf-8')
        
        logger.info(f"Extracting data from CSV: {file_path}")
        
        try:
            # Process in chunks to handle large files
            for chunk in pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                chunksize=chunk_size,
                low_memory=False
            ):
                # Convert to dict records for consistent output format
                records = chunk.to_dict('records')
                yield records
                
        except Exception as e:
            logger.error(f"Error extracting data from {file_path}: {str(e)}")
            raise
    
    def get_schema(self) -> Dict[str, str]:
        """Infer schema from CSV file."""
        file_path = self.source_config['file_path']
        
        # Read a small sample to infer types
        sample_df = pd.read_csv(file_path, nrows=10)
        return {col: str(dtype) for col, dtype in sample_df.dtypes.items()}

class S3CSVSource(DataSource):
    """Extracts data from CSV files stored in S3."""
    
    def validate_config(self) -> None:
        required_fields = ['bucket', 'key']
        missing = [f for f in required_fields if f not in self.source_config]
        if missing:
            raise ValueError(f"Missing required fields for S3CSVSource: {missing}")
    
    def extract_data(self, context: Dict[str, Any] = None) -> Iterator[Dict[str, Any]]:
        bucket = self.source_config['bucket']
        key = self.source_config['key']
        chunk_size = self.source_config.get('chunk_size', 10000)
        
        logger.info(f"Extracting data from S3: {bucket}/{key}")
        
        try:
            # Get S3 object
            s3_client = boto3.client('s3')
            response = s3_client.get_object(Bucket=bucket, Key=key)
            
            # Process CSV in chunks
            chunk = []
            reader = csv.DictReader(
                response['Body'].iter_lines(
                    chunk_size=1024,
                    decode_unicode=True
                )
            )
            
            for i, row in enumerate(reader):
                chunk.append(row)
                
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            
            # Yield any remaining records
            if chunk:
                yield chunk
                
        except Exception as e:
            logger.error(f"Error extracting from S3 {bucket}/{key}: {str(e)}")
            raise
    
    def get_schema(self) -> Dict[str, str]:
        """Sample the S3 file to infer schema."""
        bucket = self.source_config['bucket']
        key = self.source_config['key']
        
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        
        # Read a small sample
        sample_lines = []
        line_count = 0
        for line in response['Body'].iter_lines(decode_unicode=True):
            sample_lines.append(line)
            line_count += 1
            if line_count >= 11:  # header + 10 rows
                break
        
        # Parse sample with pandas
        import io
        sample_data = '\n'.join(sample_lines)
        sample_df = pd.read_csv(io.StringIO(sample_data))
        
        return {col: str(dtype) for col, dtype in sample_df.dtypes.items()}

# API-based sources
class RESTAPISource(DataSource):
    """Extracts data from REST APIs."""
    
    def validate_config(self) -> None:
        required_fields = ['base_url', 'endpoint']
        missing = [f for f in required_fields if f not in self.source_config]
        if missing:
            raise ValueError(f"Missing required fields for RESTAPISource: {missing}")
    
    def extract_data(self, context: Dict[str, Any] = None) -> Iterator[Dict[str, Any]]:
        base_url = self.source_config['base_url']
        endpoint = self.source_config['endpoint']
        headers = self.source_config.get('headers', {})
        params = self.source_config.get('params', {})
        pagination = self.source_config.get('pagination', {})
        batch_size = self.source_config.get('batch_size', 100)
        
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        logger.info(f"Extracting data from API: {url}")
        
        try:
            # Handle different pagination styles
            if pagination.get('type') == 'offset':
                offset_param = pagination.get('offset_param', 'offset')
                limit_param = pagination.get('limit_param', 'limit')
                max_pages = pagination.get('max_pages', 1000)
                
                current_params = params.copy()
                current_params[limit_param] = batch_size
                
                page = 0
                has_more = True
                
                while has_more and page < max_pages:
                    current_params[offset_param] = page * batch_size
                    
                    response = requests.get(
                        url, 
                        headers=headers, 
                        params=current_params,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Extract records based on configuration
                    if pagination.get('data_key'):
                        records = data.get(pagination['data_key'], [])
                    else:
                        records = data if isinstance(data, list) else []
                    
                    # Check if we have more pages
                    if not records:
                        has_more = False
                    
                    yield records
                    page += 1
            
            elif pagination.get('type') == 'cursor':
                cursor_param = pagination.get('cursor_param', 'cursor')
                cursor_path = pagination.get('cursor_path', 'meta.next_cursor')
                
                current_params = params.copy()
                cursor = None
                has_more = True
                
                while has_more:
                    if cursor:
                        current_params[cursor_param] = cursor
                    
                    response = requests.get(
                        url, 
                        headers=headers, 
                        params=current_params,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Extract records
                    if pagination.get('data_key'):
                        records = data.get(pagination['data_key'], [])
                    else:
                        records = data if isinstance(data, list) else []
                    
                    yield records
                    
                    # Get next cursor
                    cursor = self._get_nested_value(data, cursor_path.split('.'))
                    has_more = bool(cursor and records)
            
            else:
                # Simple one-page request
                response = requests.get(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Extract records
                if pagination.get('data_key'):
                    records = data.get(pagination['data_key'], [])
                else:
                    records = data if isinstance(data, list) else []
                
                yield records
```
## Data Engineering & ETL (continued)

```python
    def _get_nested_value(self, data, path_parts):
        """Get a nested value from a dictionary using a path."""
        current = data
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    
    def get_schema(self) -> Dict[str, str]:
        """Sample the API to infer schema."""
        # Make a small request to infer schema
        try:
            base_url = self.source_config['base_url']
            endpoint = self.source_config['endpoint']
            headers = self.source_config.get('headers', {})
            params = self.source_config.get('params', {})
            
            if 'limit' in params:
                # Ensure we only get a small sample
                params['limit'] = min(params['limit'], 5)
            elif self.source_config.get('pagination', {}).get('limit_param'):
                # Use the pagination limit parameter
                limit_param = self.source_config['pagination']['limit_param']
                params[limit_param] = 5
            
            url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract sample records
            if self.source_config.get('pagination', {}).get('data_key'):
                records = data.get(self.source_config['pagination']['data_key'], [])
            else:
                records = data if isinstance(data, list) else []
            
            if not records:
                return {}
            
            # Infer types from first record
            sample = records[0]
            return {k: type(v).__name__ for k, v in sample.items()}
            
        except Exception as e:
            logger.warning(f"Failed to infer schema from API: {str(e)}")
            return {}

# Database sources
class SQLDatabaseSource(DataSource):
    """Extracts data from SQL databases."""
    
    def validate_config(self) -> None:
        required_fields = ['connection_string', 'query']
        missing = [f for f in required_fields if f not in self.source_config]
        if missing:
            raise ValueError(f"Missing required fields for SQLDatabaseSource: {missing}")
    
    def extract_data(self, context: Dict[str, Any] = None) -> Iterator[Dict[str, Any]]:
        connection_string = self.source_config['connection_string']
        query = self.source_config['query']
        params = self.source_config.get('params', {})
        chunk_size = self.source_config.get('chunk_size', 10000)
        
        # Apply context variables to query parameters if provided
        if context and params:
            for k, v in params.items():
                if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
                    context_key = v[2:-1]
                    if context_key in context:
                        params[k] = context[context_key]
        
        logger.info(f"Extracting data from database using query: {query}")
        
        try:
            engine = create_engine(connection_string)
            
            # Use pandas to handle the chunking
            for chunk_df in pd.read_sql(
                query, 
                engine, 
                params=params, 
                chunksize=chunk_size
            ):
                yield chunk_df.to_dict('records')
                
        except Exception as e:
            logger.error(f"Database extraction error: {str(e)}")
            raise
    
    def get_schema(self) -> Dict[str, str]:
        """Query the database to get schema information."""
        connection_string = self.source_config['connection_string']
        query = self.source_config['query']
        params = self.source_config.get('params', {})
        
        try:
            engine = create_engine(connection_string)
            
            # Just fetch a few rows to infer schema
            sample_query = f"SELECT * FROM ({query}) AS sample LIMIT 5"
            sample_df = pd.read_sql(sample_query, engine, params=params)
            
            return {col: str(dtype) for col, dtype in sample_df.dtypes.items()}
            
        except Exception as e:
            logger.warning(f"Failed to infer database schema: {str(e)}")
            return {}
```

**2. Transformation Layer**

The transformation layer processes and enriches data:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Iterator, Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

class Transformer(ABC):
    """Base class for all data transformers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'unnamed_transformer')
    
    @abstractmethod
    def transform(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform a batch of records."""
        pass

class ColumnMapper(Transformer):
    """Maps columns from source format to target format."""
    
    def transform(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
            
        mapping = self.config.get('mapping', {})
        include_unmapped = self.config.get('include_unmapped', False)
        
        result = []
        for record in records:
            new_record = {}
            
            # Apply mapping
            for source_field, target_field in mapping.items():
                if source_field in record:
                    new_record[target_field] = record[source_field]
            
            # Include unmapped fields if specified
            if include_unmapped:
                for field, value in record.items():
                    if field not in mapping:
                        new_record[field] = value
            
            result.append(new_record)
        
        return result

class TypeConverter(Transformer):
    """Converts data types according to configuration."""
    
    def transform(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
            
        type_conversions = self.config.get('type_conversions', {})
        date_formats = self.config.get('date_formats', {})
        
        result = []
        for record in records:
            new_record = record.copy()
            
            for field, target_type in type_conversions.items():
                if field in new_record and new_record[field] is not None:
                    try:
                        if target_type == 'int':
                            new_record[field] = int(new_record[field])
                        elif target_type == 'float':
                            new_record[field] = float(new_record[field])
                        elif target_type == 'bool':
                            val = new_record[field]
                            if isinstance(val, str):
                                new_record[field] = val.lower() in ('true', 'yes', '1', 't', 'y')
                            else:
                                new_record[field] = bool(val)
                        elif target_type == 'str':
                            new_record[field] = str(new_record[field])
                        elif target_type == 'date' or target_type == 'datetime':
                            if isinstance(new_record[field], str):
                                format_str = date_formats.get(field, '%Y-%m-%d')
                                new_record[field] = datetime.strptime(new_record[field], format_str)
                            elif isinstance(new_record[field], (int, float)):
                                # Assume Unix timestamp
                                new_record[field] = datetime.fromtimestamp(new_record[field])
                    except Exception as e:
                        logger.warning(f"Type conversion error for field {field}: {str(e)}")
            
            result.append(new_record)
        
        return result

class DataCleaner(Transformer):
    """Cleans data by handling missing values, duplicates, etc."""
    
    def transform(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
            
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(records)
        
        # Handle missing values
        missing_strategy = self.config.get('missing_strategy', {})
        for column, strategy in missing_strategy.items():
            if column in df.columns:
                if strategy == 'drop':
                    df = df.dropna(subset=[column])
                elif strategy == 'fill_zero':
                    df[column] = df[column].fillna(0)
                elif strategy == 'fill_empty_string':
                    df[column] = df[column].fillna('')
                elif strategy == 'fill_mean' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].mean())
                elif strategy == 'fill_median' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].median())
                elif isinstance(strategy, (int, float, str)):
                    df[column] = df[column].fillna(strategy)
        
        # Remove duplicates if specified
        if self.config.get('remove_duplicates', False):
            subset = self.config.get('duplicate_subset', None)
            df = df.drop_duplicates(subset=subset)
        
        # Apply filtering if specified
        filters = self.config.get('filters', [])
        for filter_config in filters:
            column = filter_config.get('column')
            operator = filter_config.get('operator')
            value = filter_config.get('value')
            
            if column and operator and value is not None and column in df.columns:
                if operator == '==':
                    df = df[df[column] == value]
                elif operator == '!=':
                    df = df[df[column] != value]
                elif operator == '>':
                    df = df[df[column] > value]
                elif operator == '>=':
                    df = df[df[column] >= value]
                elif operator == '<':
                    df = df[df[column] < value]
                elif operator == '<=':
                    df = df[df[column] <= value]
                elif operator == 'in' and isinstance(value, list):
                    df = df[df[column].isin(value)]
                elif operator == 'not_in' and isinstance(value, list):
                    df = df[~df[column].isin(value)]
        
        # Convert back to records
        return df.to_dict('records')

class DataEnricher(Transformer):
    """Enriches data with additional fields or lookups."""
    
    def transform(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
            
        # Add computed fields
        computed_fields = self.config.get('computed_fields', {})
        
        result = []
        for record in records:
            new_record = record.copy()
            
            # Add computed fields
            for field, expression in computed_fields.items():
                try:
                    # Simple expressions
                    if expression.get('type') == 'concatenate':
                        source_fields = expression.get('source_fields', [])
                        separator = expression.get('separator', '')
                        values = [str(new_record.get(f, '')) for f in source_fields]
                        new_record[field] = separator.join(values)
                    
                    elif expression.get('type') == 'arithmetic':
                        operation = expression.get('operation')
                        operands = expression.get('operands', [])
                        
                        # Get values of operands
                        values = []
                        for operand in operands:
                            if isinstance(operand, str) and operand in new_record:
                                values.append(new_record[operand])
                            else:
                                values.append(operand)
                        
                        # Perform operation
                        if operation == 'add' and len(values) >= 2:
                            new_record[field] = values[0] + values[1]
                        elif operation == 'subtract' and len(values) >= 2:
                            new_record[field] = values[0] - values[1]
                        elif operation == 'multiply' and len(values) >= 2:
                            new_record[field] = values[0] * values[1]
                        elif operation == 'divide' and len(values) >= 2 and values[1] != 0:
                            new_record[field] = values[0] / values[1]
                    
                    elif expression.get('type') == 'current_time':
                        format_str = expression.get('format')
                        now = datetime.now(timezone.utc)
                        if format_str:
                            new_record[field] = now.strftime(format_str)
                        else:
                            new_record[field] = now
                    
                    elif expression.get('type') == 'hash':
                        source_fields = expression.get('source_fields', [])
                        algorithm = expression.get('algorithm', 'md5')
                        
                        # Concatenate values
                        values = [str(new_record.get(f, '')) for f in source_fields]
                        concatenated = '|'.join(values)
                        
                        # Hash using specified algorithm
                        if algorithm == 'md5':
                            new_record[field] = hashlib.md5(concatenated.encode()).hexdigest()
                        elif algorithm == 'sha1':
                            new_record[field] = hashlib.sha1(concatenated.encode()).hexdigest()
                        elif algorithm == 'sha256':
                            new_record[field] = hashlib.sha256(concatenated.encode()).hexdigest()
                
                except Exception as e:
                    logger.warning(f"Error computing field {field}: {str(e)}")
            
            result.append(new_record)
        
        return result

class TransformationPipeline:
    """Orchestrates multiple transformers in sequence."""
    
    def __init__(self, transformers: List[Transformer]):
        self.transformers = transformers
    
    def process(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all transformers in sequence."""
        current_records = records
        
        for transformer in self.transformers:
            try:
                logger.debug(f"Applying transformer: {transformer.name}")
                current_records = transformer.transform(current_records)
            except Exception as e:
                logger.error(f"Error in transformer {transformer.name}: {str(e)}")
                raise
        
        return current_records
```

**3. Loading Layer**

The loading layer writes processed data to destinations:

```python
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import psycopg2
import pymysql
import sqlalchemy
from sqlalchemy import create_engine
import boto3
import json
import csv
import os
from io import StringIO
from datetime import datetime

logger = logging.getLogger(__name__)

class DataDestination(ABC):
    """Abstract base class for all data destinations."""
    
    def __init__(self, dest_config: Dict[str, Any]):
        self.dest_config = dest_config
        self.name = dest_config.get('name', 'unnamed_destination')
    
    @abstractmethod
    def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write a batch of records to the destination."""
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """Perform any finalization steps after all batches are written."""
        pass

class PostgresDestination(DataDestination):
    """Writes data to a PostgreSQL database."""
    
    def __init__(self, dest_config: Dict[str, Any]):
        super().__init__(dest_config)
        self.connection_string = dest_config.get('connection_string')
        self.table = dest_config.get('table')
        self.schema = dest_config.get('schema', 'public')
        self.if_exists = dest_config.get('if_exists', 'append')
        
        # Connect to database
        self.engine = create_engine(self.connection_string)
    
    def write_batch(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        
        df = pd.DataFrame(records)
        
        # Write to PostgreSQL
        table_name = f"{self.schema}.{self.table}" if self.schema else self.table
        df.to_sql(
            self.table,
            self.engine,
            schema=self.schema,
            if_exists=self.if_exists,
            index=False,
            chunksize=1000
        )
        
        logger.info(f"Wrote {len(records)} records to Postgres table {table_name}")
    
    def finalize(self) -> None:
        # Close connections
        self.engine.dispose()
        logger.info(f"Closed connection to Postgres destination: {self.name}")

class S3Destination(DataDestination):
    """Writes data to Amazon S3 in various formats."""
    
    def __init__(self, dest_config: Dict[str, Any]):
        super().__init__(dest_config)
        self.bucket = dest_config.get('bucket')
        self.key_prefix = dest_config.get('key_prefix', '')
        self.format = dest_config.get('format', 'csv').lower()
        self.partitioning = dest_config.get('partitioning', {})
        self.s3_client = boto3.client('s3')
        
        # For collecting data when using non-partitioned mode
        self.collected_records = []
    
    def write_batch(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        
        # If partitioning is enabled, write each batch according to partitioning
        if self.partitioning:
            self._write_partitioned_batch(records)
        else:
            # Collect records for writing in finalize
            self.collected_records.extend(records)
    
    def _write_partitioned_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write records with partitioning."""
        partition_columns = self.partitioning.get('columns', [])
        date_format = self.partitioning.get('date_format', '%Y/%m/%d')
        
        # Group records by partition values
        partition_groups = {}
        for record in records:
            # Build partition key
            partition_key_parts = []
            for col in partition_columns:
                if col in record:
                    value = record[col]
                    # Format dates according to configuration
                    if isinstance(value, datetime):
                        value = value.strftime(date_format)
                    partition_key_parts.append(f"{col}={value}")
            
            partition_key = '/'.join(partition_key_parts)
            
            if partition_key not in partition_groups:
                partition_groups[partition_key] = []
            
            partition_groups[partition_key].append(record)
        
        # Write each partition
        for partition_key, partition_records in partition_groups.items():
            # Build full key including prefix and partition
            if partition_key:
                full_key = f"{self.key_prefix.rstrip('/')}/{partition_key}/data.{self.format}"
            else:
                full_key = f"{self.key_prefix.rstrip('/')}/data.{self.format}"
            
            # Write the partition
            self._write_to_s3(partition_records, full_key)
    
    def _write_to_s3(self, records: List[Dict[str, Any]], key: str) -> None:
        """Write records to S3 in the specified format."""
        if not records:
            return
        
        # Convert to appropriate format
        if self.format == 'csv':
            # Convert to CSV
            df = pd.DataFrame(records)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            file_content = csv_buffer.getvalue()
            content_type = 'text/csv'
        
        elif self.format == 'json':
            # Convert to JSON
            file_content = json.dumps(records, default=str)
            content_type = 'application/json'
        
        elif self.format == 'parquet':
            # Convert to Parquet
            import io
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            df = pd.DataFrame(records)
            buffer = io.BytesIO()
            
            # Convert to pyarrow table and write as parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(table, buffer)
            
            # Get the content and reset buffer
            buffer.seek(0)
            file_content = buffer.getvalue()
            content_type = 'application/octet-stream'
        
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=file_content,
            ContentType=content_type
        )
        
        logger.info(f"Wrote {len(records)} records to S3: s3://{self.bucket}/{key}")
    
    def finalize(self) -> None:
        """Write any remaining data in non-partitioned mode."""
        if not self.partitioning and self.collected_records:
            # Generate a timestamp-based key
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            key = f"{self.key_prefix.rstrip('/')}/data_{timestamp}.{self.format}"
            
            # Write all collected records
            self._write_to_s3(self.collected_records, key)
            self.collected_records = []
        
        logger.info(f"Finalized S3 destination: {self.name}")

class BigQueryDestination(DataDestination):
    """Writes data to Google BigQuery."""
    
    def __init__(self, dest_config: Dict[str, Any]):
        super().__init__(dest_config)
        self.project_id = dest_config.get('project_id')
        self.dataset_id = dest_config.get('dataset_id')
        self.table_id = dest_config.get('table_id')
        self.if_exists = dest_config.get('if_exists', 'append')
        
        # Import here to avoid dependency if not used
        from google.cloud import bigquery
        self.client = bigquery.Client(project=self.project_id)
        
        # For collecting records if needed
        self.collected_records = []
    
    def write_batch(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        
        # Collect records for later processing
        self.collected_records.extend(records)
        
        # If batch size exceeds threshold, write immediately
        batch_size = self.dest_config.get('batch_size', 10000)
        if len(self.collected_records) >= batch_size:
            self._write_to_bigquery(self.collected_records)
            self.collected_records = []
    
    def _write_to_bigquery(self, records: List[Dict[str, Any]]) -> None:
        """Write records to BigQuery."""
        if not records:
            return
        
        from google.cloud import bigquery
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Determine table reference
        table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
        
        # Map pandas dtypes to BigQuery schema
        type_mapping = {
            'int64': 'INT64',
            'float64': 'FLOAT64',
            'bool': 'BOOL',
            'datetime64[ns]': 'TIMESTAMP',
            'object': 'STRING'
        }
        
        # Convert to schema if needed
        schema = []
        for col, dtype in df.dtypes.items():
            bq_type = type_mapping.get(str(dtype), 'STRING')
            field = bigquery.SchemaField(col, bq_type)
            schema.append(field)
        
        # Determine job config
        job_config = bigquery.LoadJobConfig()
        
        if self.if_exists == 'replace':
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        else:
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        
        job_config.schema = schema
        
        # Load data
        job = self.client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
        
        # Wait for the job to complete
        job.result()
        
        logger.info(f"Wrote {len(records)} records to BigQuery table: {table_ref}")
    
    def finalize(self) -> None:
        """Write any remaining records."""
        if self.collected_records:
            self._write_to_bigquery(self.collected_records)
            self.collected_records = []
```

**4. Pipeline Orchestration**

The orchestration layer ties everything together with error handling and monitoring:

```python
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import concurrent.futures
from datetime import datetime, timedelta
import json
import os
import uuid

from extraction import DataSource
from transformation import TransformationPipeline, Transformer
from loading import DataDestination

logger = logging.getLogger(__name__)

@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""
    start_time: datetime
    end_time: Optional[datetime] = None
    records_read: int = 0
    records_written: int = 0
    errors: int = 0
    warnings: int = 0
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate pipeline duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "records_read": self.records_read,
            "records_written": self.records_written,
            "errors": self.errors,
            "warnings": self.warnings
        }

class DataPipeline:
    """Main pipeline class that orchestrates the ETL process."""
    
    def __init__(
        self, 
        source: DataSource, 
        transformation_pipeline: TransformationPipeline,
        destinations: List[DataDestination],
        batch_size: int = 10000,
        max_retries: int = 3,
        retry_delay: int = 5,
        parallel_destinations: bool = True
    ):
        self.source = source
        self.transformation_pipeline = transformation_pipeline
        self.destinations = destinations
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.parallel_destinations = parallel_destinations
        self.stats = PipelineStats(start_time=datetime.now())
        self.pipeline_id = str(uuid.uuid4())
    
    def run(self, context: Dict[str, Any] = None) -> PipelineStats:
        """Run the complete pipeline."""
        context = context or {}
        logger.info(f"Starting pipeline {self.pipeline_id} with source: {self.source.name}")
        
        try:
            # Extract and process data in batches
            for batch_num, records_batch in enumerate(self.source.extract_data(context)):
                logger.info(f"Processing batch {batch_num+1} with {len(records_batch)} records")
                
                # Update stats
                self.stats.records_read += len(records_batch)
                
                # Transform data
                try:
                    transformed_records = self.transformation_pipeline.process(records_batch)
                    logger.info(f"Transformed batch {batch_num+1}: {len(transformed_records)} records")
                except Exception as e:
                    logger.error(f"Error transforming batch {batch_num+1}: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.stats.errors += 1
                    continue
                
                # Load data to all destinations
                if self.parallel_destinations and len(self.destinations) > 1:
                    self._write_to_destinations_parallel(transformed_records)
                else:
                    self._write_to_destinations_sequential(transformed_records)
            
            # Finalize all destinations
            for destination in self.destinations:
                try:
                    destination.finalize()
                except Exception as e:
                    logger.error(f"Error finalizing destination {destination.name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.stats.errors += 1
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            logger.error(traceback.format_exc())
            self.stats.errors += 1
        
        finally:
            # Record end time
            self.stats.end_time = datetime.now()
            
            # Log pipeline completion
            duration = self.stats.duration_seconds
            logger.info(
                f"Pipeline {self.pipeline_id} completed in {duration:.2f} seconds. "
                f"Records read: {self.stats.records_read}, "
                f"Records written: {self.stats.records_written}, "
                f"Errors: {self.stats.errors}"
            )
            
            return self.stats
    
    def _write_to_destinations_sequential(self, records: List[Dict[str, Any]]) -> None:
        """Write records to all destinations sequentially."""
        for destination in self.destinations:
            self._write_with_retry(destination, records)
    
    def _write_to_destinations_parallel(self, records: List[Dict[str, Any]]) -> None:
        """Write records to all destinations in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.destinations)) as executor:
            futures = {
                executor.submit(self._write_with_retry, destination, records): destination
                for destination in self.destinations
            }
            
            for future in concurrent.futures.as_completed(futures):
                destination = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in parallel write to {destination.name}: {str(e)}")
                    self.stats.errors += 1
    
    def _write_with_retry(self, destination: DataDestination, records: List[Dict[str, Any]]) -> None:
        """Write records to a destination with retry logic."""
        retries = 0
        while retries <= self.max_retries:
            try:
                destination.write_batch(records)
                self.stats.records_written += len(records)
                break
            except Exception as e:
                retries += 1
                logger.warning(
                    f"Error writing to destination {destination
```
## Data Engineering & ETL (continued)

```python
    def _write_with_retry(self, destination: DataDestination, records: List[Dict[str, Any]]) -> None:
        """Write records to a destination with retry logic."""
        retries = 0
        while retries <= self.max_retries:
            try:
                destination.write_batch(records)
                self.stats.records_written += len(records)
                break
            except Exception as e:
                retries += 1
                logger.warning(
                    f"Error writing to destination {destination.name}, attempt {retries}/{self.max_retries}: {str(e)}"
                )
                self.stats.warnings += 1
                
                if retries > self.max_retries:
                    logger.error(f"Failed to write to destination {destination.name} after {self.max_retries} attempts")
                    self.stats.errors += 1
                    raise
                
                # Wait before retrying
                time.sleep(self.retry_delay * retries)  # Exponential backoff
```

**5. Monitoring and Error Handling**

For comprehensive monitoring, I'd implement a monitoring service:

```python
import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import threading
from queue import Queue
import os
import socket
import traceback
import boto3
import prometheus_client as prom
import requests

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and exports pipeline metrics."""
    
    def __init__(self, app_name: str, export_prometheus: bool = True):
        self.app_name = app_name
        
        # Prometheus metrics
        if export_prometheus:
            # Counter metrics
            self.records_processed = prom.Counter(
                f'{app_name}_records_processed_total',
                'Total number of records processed',
                ['pipeline_id', 'source']
            )
            
            self.records_written = prom.Counter(
                f'{app_name}_records_written_total',
                'Total number of records written to destinations',
                ['pipeline_id', 'destination']
            )
            
            self.errors = prom.Counter(
                f'{app_name}_errors_total',
                'Total number of errors',
                ['pipeline_id', 'component', 'error_type']
            )
            
            # Gauge metrics
            self.pipeline_duration = prom.Gauge(
                f'{app_name}_pipeline_duration_seconds',
                'Duration of pipeline execution in seconds',
                ['pipeline_id']
            )
            
            self.last_run_timestamp = prom.Gauge(
                f'{app_name}_last_run_timestamp',
                'Timestamp of the last pipeline run',
                ['pipeline_id', 'status']
            )
            
            # Histogram metrics
            self.batch_processing_time = prom.Histogram(
                f'{app_name}_batch_processing_time_seconds',
                'Time to process a batch of records',
                ['pipeline_id', 'stage'],
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
            )
            
            # Start prometheus HTTP server if not already running
            if not hasattr(prom, '_server_started'):
                prom.start_http_server(8000)
                prom._server_started = True
    
    def record_processed(self, pipeline_id: str, source: str, count: int) -> None:
        """Record processed records count."""
        self.records_processed.labels(pipeline_id=pipeline_id, source=source).inc(count)
    
    def record_written(self, pipeline_id: str, destination: str, count: int) -> None:
        """Record written records count."""
        self.records_written.labels(pipeline_id=pipeline_id, destination=destination).inc(count)
    
    def record_error(self, pipeline_id: str, component: str, error_type: str) -> None:
        """Record an error."""
        self.errors.labels(pipeline_id=pipeline_id, component=component, error_type=error_type).inc()
    
    def record_pipeline_duration(self, pipeline_id: str, duration: float) -> None:
        """Record pipeline duration."""
        self.pipeline_duration.labels(pipeline_id=pipeline_id).set(duration)
    
    def record_pipeline_completion(self, pipeline_id: str, status: str) -> None:
        """Record pipeline completion."""
        self.last_run_timestamp.labels(pipeline_id=pipeline_id, status=status).set_to_current_time()
    
    def time_batch_processing(self, pipeline_id: str, stage: str):
        """Context manager to time batch processing."""
        return self.batch_processing_time.labels(pipeline_id=pipeline_id, stage=stage).time()

class AlertManager:
    """Manages alerting for pipeline errors and warnings."""
    
    def __init__(
        self,
        app_name: str,
        enable_email: bool = False,
        enable_slack: bool = False,
        enable_pagerduty: bool = False,
        alert_config: Dict[str, Any] = None
    ):
        self.app_name = app_name
        self.host = socket.gethostname()
        self.enable_email = enable_email
        self.enable_slack = enable_slack
        self.enable_pagerduty = enable_pagerduty
        self.alert_config = alert_config or {}
        
        # Set up alert queues and start alert worker threads
        self.alert_queue = Queue()
        
        # Start alert worker thread
        self.alert_worker = threading.Thread(target=self._process_alerts, daemon=True)
        self.alert_worker.start()
    
    def alert(
        self,
        level: str,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        pipeline_id: Optional[str] = None
    ) -> None:
        """Queue an alert to be sent."""
        alert_data = {
            "level": level,
            "title": title,
            "message": message,
            "details": details or {},
            "pipeline_id": pipeline_id,
            "timestamp": datetime.now().isoformat(),
            "host": self.host,
            "app": self.app_name
        }
        
        # Add to queue for async processing
        self.alert_queue.put(alert_data)
        
        # Also log the alert
        log_message = f"ALERT [{level}] {title}: {message}"
        if level.lower() == "critical":
            logger.critical(log_message)
        elif level.lower() == "error":
            logger.error(log_message)
        elif level.lower() == "warning":
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _process_alerts(self) -> None:
        """Background worker to process and send alerts."""
        while True:
            try:
                # Get alert from queue
                alert_data = self.alert_queue.get()
                
                # Send alerts through configured channels
                level = alert_data["level"].lower()
                
                # Only send higher level alerts to certain channels
                if level in ("critical", "error"):
                    if self.enable_email:
                        self._send_email_alert(alert_data)
                    
                    if self.enable_pagerduty:
                        self._send_pagerduty_alert(alert_data)
                
                # Send all alerts to Slack
                if self.enable_slack:
                    self._send_slack_alert(alert_data)
                
                self.alert_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error processing alert: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Sleep briefly before continuing
                time.sleep(1)
    
    def _send_email_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send an email alert."""
        if "email" not in self.alert_config:
            logger.warning("Email alerting enabled but no configuration provided")
            return
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            config = self.alert_config.get("email", {})
            
            # Prepare email
            msg = MIMEMultipart()
            msg['From'] = config.get("from_address", "alerts@example.com")
            msg['To'] = ", ".join(config.get("to_addresses", []))
            msg['Subject'] = f"[{alert_data['level'].upper()}] {self.app_name}: {alert_data['title']}"
            
            # Format message body
            body = f"""
            <h2>{alert_data['title']}</h2>
            <p><strong>Level:</strong> {alert_data['level']}</p>
            <p><strong>Time:</strong> {alert_data['timestamp']}</p>
            <p><strong>Host:</strong> {alert_data['host']}</p>
            <p><strong>Application:</strong> {alert_data['app']}</p>
            <p><strong>Pipeline ID:</strong> {alert_data.get('pipeline_id', 'N/A')}</p>
            <p><strong>Message:</strong> {alert_data['message']}</p>
            
            <h3>Details:</h3>
            <pre>{json.dumps(alert_data['details'], indent=2)}</pre>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(config.get("smtp_server"), config.get("smtp_port", 25)) as server:
                if config.get("use_tls", False):
                    server.starttls()
                
                if "username" in config and "password" in config:
                    server.login(config["username"], config["password"])
                
                server.send_message(msg)
            
            logger.info(f"Sent email alert: {alert_data['title']}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
    
    def _send_slack_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send a Slack alert."""
        if "slack" not in self.alert_config:
            logger.warning("Slack alerting enabled but no configuration provided")
            return
        
        try:
            config = self.alert_config.get("slack", {})
            webhook_url = config.get("webhook_url")
            
            if not webhook_url:
                logger.warning("No Slack webhook URL configured")
                return
            
            # Determine color based on level
            color = {
                "info": "#36a64f",  # green
                "warning": "#ffcc00",  # yellow
                "error": "#ff9900",  # orange
                "critical": "#ff0000"  # red
            }.get(alert_data["level"].lower(), "#36a64f")
            
            # Format message
            message = {
                "attachments": [
                    {
                        "fallback": f"{alert_data['level'].upper()}: {alert_data['title']}",
                        "color": color,
                        "title": f"{alert_data['title']}",
                        "text": alert_data['message'],
                        "fields": [
                            {
                                "title": "Level",
                                "value": alert_data['level'].upper(),
                                "short": True
                            },
                            {
                                "title": "Application",
                                "value": alert_data['app'],
                                "short": True
                            },
                            {
                                "title": "Host",
                                "value": alert_data['host'],
                                "short": True
                            },
                            {
                                "title": "Pipeline ID",
                                "value": alert_data.get('pipeline_id', 'N/A'),
                                "short": True
                            }
                        ],
                        "footer": f"Time: {alert_data['timestamp']}",
                    }
                ]
            }
            
            # Add details if they exist and aren't too large
            details_str = json.dumps(alert_data['details'], indent=2)
            if details_str and len(details_str) < 1000:  # Slack limits attachment size
                message["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": f"```{details_str}```",
                    "short": False
                })
            
            # Send to Slack
            response = requests.post(
                webhook_url,
                json=message,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Sent Slack alert: {alert_data['title']}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
    
    def _send_pagerduty_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send a PagerDuty alert."""
        if "pagerduty" not in self.alert_config:
            logger.warning("PagerDuty alerting enabled but no configuration provided")
            return
        
        try:
            config = self.alert_config.get("pagerduty", {})
            integration_key = config.get("integration_key")
            
            if not integration_key:
                logger.warning("No PagerDuty integration key configured")
                return
            
            # Map alert level to PagerDuty severity
            severity = {
                "warning": "warning",
                "error": "error",
                "critical": "critical"
            }.get(alert_data["level"].lower(), "info")
            
            # Create unique incident key
            incident_key = f"{self.app_name}-{alert_data.get('pipeline_id', 'unknown')}-{int(time.time())}"
            
            # Format event
            event = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "dedup_key": incident_key,
                "payload": {
                    "summary": f"{alert_data['title']}: {alert_data['message']}",
                    "source": alert_data['host'],
                    "severity": severity,
                    "component": self.app_name,
                    "group": "pipeline",
                    "custom_details": alert_data['details']
                }
            }
            
            # Send to PagerDuty
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=event,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Sent PagerDuty alert: {alert_data['title']}")
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {str(e)}")

class PipelineMonitor:
    """Main monitoring class that integrates metrics and alerts."""
    
    def __init__(
        self,
        app_name: str,
        enable_metrics: bool = True,
        enable_alerts: bool = True,
        alert_config: Dict[str, Any] = None
    ):
        self.app_name = app_name
        
        # Initialize components
        self.metrics = MetricsCollector(app_name) if enable_metrics else None
        
        self.alert_manager = AlertManager(
            app_name,
            enable_email=alert_config.get("enable_email", False),
            enable_slack=alert_config.get("enable_slack", False),
            enable_pagerduty=alert_config.get("enable_pagerduty", False),
            alert_config=alert_config
        ) if enable_alerts else None
    
    def record_pipeline_start(self, pipeline_id: str, source_name: str) -> None:
        """Record pipeline start event."""
        logger.info(f"Pipeline {pipeline_id} started with source: {source_name}")
        
        if self.metrics:
            # Reset counters for new run
            self.metrics.record_pipeline_completion(pipeline_id, "started")
    
    def record_pipeline_end(
        self, 
        pipeline_id: str, 
        stats: Dict[str, Any],
        success: bool
    ) -> None:
        """Record pipeline end event."""
        status = "success" if success else "failed"
        duration = stats.get("duration_seconds", 0)
        
        logger.info(
            f"Pipeline {pipeline_id} completed with status: {status} in {duration:.2f} seconds. "
            f"Records read: {stats.get('records_read', 0)}, "
            f"Records written: {stats.get('records_written', 0)}, "
            f"Errors: {stats.get('errors', 0)}"
        )
        
        if self.metrics:
            self.metrics.record_pipeline_duration(pipeline_id, duration)
            self.metrics.record_pipeline_completion(pipeline_id, status)
        
        # Send alert for failed pipelines
        if not success and self.alert_manager:
            self.alert_manager.alert(
                level="error",
                title=f"Pipeline {pipeline_id} failed",
                message=f"Pipeline failed with {stats.get('errors', 0)} errors",
                details=stats,
                pipeline_id=pipeline_id
            )
    
    def record_batch_processed(
        self, 
        pipeline_id: str, 
        source_name: str, 
        destination_name: str,
        batch_num: int,
        records_count: int
    ) -> None:
        """Record batch processing event."""
        logger.info(
            f"Pipeline {pipeline_id}: Processed batch {batch_num} with {records_count} records "
            f"from {source_name} to {destination_name}"
        )
        
        if self.metrics:
            self.metrics.record_processed(pipeline_id, source_name, records_count)
            self.metrics.record_written(pipeline_id, destination_name, records_count)
    
    def record_error(
        self,
        pipeline_id: str,
        component: str,
        error_type: str,
        error_message: str,
        details: Dict[str, Any] = None
    ) -> None:
        """Record an error event."""
        logger.error(
            f"Pipeline {pipeline_id}: Error in {component} - {error_type}: {error_message}"
        )
        
        if self.metrics:
            self.metrics.record_error(pipeline_id, component, error_type)
        
        # Only alert on critical errors
        critical_errors = ["connection_failure", "data_corruption", "permission_denied"]
        
        if self.alert_manager and error_type in critical_errors:
            self.alert_manager.alert(
                level="error",
                title=f"Pipeline error in {component}",
                message=error_message,
                details=details or {},
                pipeline_id=pipeline_id
            )
    
    def time_operation(self, pipeline_id: str, stage: str):
        """Time a pipeline operation."""
        if self.metrics:
            return self.metrics.time_batch_processing(pipeline_id, stage)
        else:
            # Return dummy context manager if metrics disabled
            class DummyContextManager:
                def __enter__(self):
                    return None
                def __exit__(self, *args):
                    pass
            return DummyContextManager()
```

**6. Main Application**

Finally, a main application to bring it all together:

```python
import logging
import argparse
import json
import sys
import os
from typing import Dict, Any, List
import yaml
from datetime import datetime, timedelta
import uuid
import time
import traceback

# Import our components
from extraction import (
    CSVSource, S3CSVSource, RESTAPISource, SQLDatabaseSource, DataSource
)
from transformation import (
    ColumnMapper, TypeConverter, DataCleaner, DataEnricher,
    TransformationPipeline, Transformer
)
from loading import (
    PostgresDestination, S3Destination, BigQueryDestination, DataDestination
)
from pipeline import DataPipeline, PipelineStats
from monitoring import PipelineMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    return config

def create_source(source_config: Dict[str, Any]) -> DataSource:
    """Create a data source from configuration."""
    source_type = source_config.get('type')
    
    if source_type == 'csv':
        return CSVSource(source_config)
    elif source_type == 's3_csv':
        return S3CSVSource(source_config)
    elif source_type == 'rest_api':
        return RESTAPISource(source_config)
    elif source_type == 'sql_database':
        return SQLDatabaseSource(source_config)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

def create_transformers(transformers_config: List[Dict[str, Any]]) -> List[Transformer]:
    """Create transformation steps from configuration."""
    transformers = []
    
    for config in transformers_config:
        transformer_type = config.get('type')
        
        if transformer_type == 'column_mapper':
            transformers.append(ColumnMapper(config))
        elif transformer_type == 'type_converter':
            transformers.append(TypeConverter(config))
        elif transformer_type == 'data_cleaner':
            transformers.append(DataCleaner(config))
        elif transformer_type == 'data_enricher':
            transformers.append(DataEnricher(config))
        else:
            raise ValueError(f"Unsupported transformer type: {transformer_type}")
    
    return transformers

def create_destinations(destinations_config: List[Dict[str, Any]]) -> List[DataDestination]:
    """Create data destinations from configuration."""
    destinations = []
    
    for config in destinations_config:
        dest_type = config.get('type')
        
        if dest_type == 'postgres':
            destinations.append(PostgresDestination(config))
        elif dest_type == 's3':
            destinations.append(S3Destination(config))
        elif dest_type == 'bigquery':
            destinations.append(BigQueryDestination(config))
        else:
            raise ValueError(f"Unsupported destination type: {dest_type}")
    
    return destinations

def run_pipeline(config_path: str, context: Dict[str, Any] = None) -> PipelineStats:
    """Run a data pipeline from configuration."""
    # Generate pipeline ID
    pipeline_id = str(uuid.uuid4())
    
    # Create monitor
    monitor = PipelineMonitor(
        app_name="data_pipeline",
        enable_metrics=True,
        enable_alerts=True,
        alert_config={
            "enable_slack": True,
            "slack": {
                "webhook_url": os.environ.get("SLACK_WEBHOOK_URL")
            }
        }
    )
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Create source
        source = create_source(config.get('source', {}))
        
        # Create transformers
        transformers = create_transformers(config.get('transformers', []))
        transformation_pipeline = TransformationPipeline(transformers)
        
        # Create destinations
        destinations = create_destinations(config.get('destinations', []))
        
        # Prepare context
        pipeline_context = context or {}
        # Add default context variables
        pipeline_context.update({
            "execution_date": datetime.now().strftime('%Y-%m-%d'),
            "execution_timestamp": int(datetime.now().timestamp()),
            "pipeline_id": pipeline_id
        })
        
        # Record pipeline start
        monitor.record_pipeline_start(pipeline_id, source.name)
        
        # Create pipeline
        pipeline = DataPipeline(
            source=source,
            transformation_pipeline=transformation_pipeline,
            destinations=destinations,
            batch_size=config.get('batch_size', 10000),
            max_retries=config.get('max_retries', 3),
            retry_delay=config.get('retry_delay', 5),
            parallel_destinations=config.get('parallel_destinations', True)
        )
        
        # Run pipeline
        stats = pipeline.run(context=pipeline_context)
        
        # Record completion
        monitor.record_pipeline_end(
            pipeline_id=pipeline_id,
            stats=stats.to_dict(),
            success=(stats.errors == 0)
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Record pipeline failure
        monitor.record_error(
            pipeline_id=pipeline_id,
            component="main",
            error_type="pipeline_initialization",
            error_message=str(e),
            details={"traceback": traceback.format_exc()}
        )
        
        monitor.record_pipeline_end(
            pipeline_id=pipeline_id,
            stats={"errors": 1, "duration_seconds": 0},
            success=False
        )
        
        # Re-raise to handle at higher level
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a data pipeline from configuration')
    parser.add_argument('config', help='Path to pipeline configuration file (YAML or JSON)')
    args = parser.parse_args()
    
    try:
        stats = run_pipeline(args.config)
        
        if stats.errors > 0:
            sys.exit(1)
            
    except Exception:
        sys.exit(1)
```

**7. Example Pipeline Configuration**

Here's an example configuration for a real-world pipeline:

```yaml
# sales_pipeline.yaml
name: sales_data_pipeline
description: "Processes daily sales data from CSV files, API, and databases"

# Source configuration
source:
  type: s3_csv
  name: sales_data
  bucket: sales-data-bucket
  key: daily/sales_${execution_date}.csv
  chunk_size: 50000

# Transformation steps
transformers:
  - type: column_mapper
    name: rename_columns
    mapping:
      transaction_id: sale_id
      customer_id: customer_id
      product_id: product_id
      transaction_date: sale_date
      quantity: quantity
      price: unit_price
    include_unmapped: false
  
  - type: type_converter
    name: convert_data_types
    type_conversions:
      sale_id: str
      customer_id: str
      product_id: str
      sale_date: date
      quantity: int
      unit_price: float
    date_formats:
      sale_date: "%Y-%m-%d"
  
  - type: data_cleaner
    name: clean_sales_data
    missing_strategy:
      customer_id: fill_empty_string
      quantity: fill_zero
      unit_price: fill_zero
    remove_duplicates: true
    duplicate_subset:
      - sale_id
    filters:
      - column: unit_price
        operator: ">"
        value: 0
  
  - type: data_enricher
    name: enrich_sales_data
    computed_fields:
      total_amount:
        type: arithmetic
        operation: multiply
        operands:
          - quantity
          - unit_price
      
      processing_date:
        type: current_time
        format: "%Y-%m-%d"
      
      sale_id_hash:
        type: hash
        algorithm: sha256
        source_fields:
          - sale_id
          - customer_id

# Destinations
destinations:
  - type: postgres
    name: sales_analytics_db
    connection_string: postgresql://user:password@postgres-host:5432/analytics
    table: sales_data
    schema: public
    if_exists: append
  
  - type: s3
    name: sales_data_lake
    bucket: sales-data-lake
    key_prefix: processed/sales/${execution_date}
    format: parquet
    partitioning:
      columns:
        - sale_date
      date_format: "%Y/%m/%d"

# Pipeline settings
batch_size: 25000
max_retries: 5
retry_delay: 10
parallel_destinations: true
```

**Scaling Considerations:**

1. **Horizontal Scaling**
    
    - Deploy multiple pipeline instances to process partitioned data
    - Use Kubernetes for dynamic scaling based on workload
    - Implement partitioning by date, ID ranges, or geographic regions
2. **Vertical Scaling**
    
    - Configure memory settings for optimal performance
    - Use chunking to process data in manageable batches
    - Optimize Python code with efficient libraries (pandas, numpy)
3. **Resource Optimization**
    
    - Implement backpressure mechanisms to prevent overwhelming resources
    - Use connection pooling for database operations
    - Leverage batch operations to reduce network overhead
4. **Distributed Processing**
    
    - For larger workloads, integrate with Apache Spark or Dask
    - Use cloud services like AWS Glue or GCP Dataflow
    - Implement work queues with Redis or RabbitMQ
5. **Production Deployment**
    
    ```yaml
    # docker-compose.yml for local deployment
    version: '3'
    services:
      pipeline:
        build: .
        volumes:
          - ./config:/app/config
          - ./logs:/app/logs
        environment:
          - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
          - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
          - SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}
        command: python main.py /app/config/sales_pipeline.yaml
    
      prometheus:
        image: prom/prometheus
        ports:
          - "9090:9090"
        volumes:
          - ./prometheus:/etc/prometheus
    
      grafana:
        image: grafana/grafana
        ports:
          - "3000:3000"
        volumes:
          - ./grafana:/var/lib/grafana
        depends_on:
          - prometheus
    ```
    

**Error Handling and Resilience:**

1. **Granular Error Management**
    
    - Different strategies for different error types
    - Transient errors (network, timeouts) → Retry with exponential backoff
    - Data errors → Log and continue or redirect to error queue
    - Critical errors → Alert and halt
2. **Data Quality Checks**
    
    - Schema validation before processing
    - Statistical checks for anomaly detection
    - Data type validation and constraint enforcement
3. **Circuit Breakers**
    
    - Prevent cascading failures by detecting when services are down
    - Fail fast when dependencies are unavailable
    - Gradual recovery with canary testing
4. **Dead Letter Queues**
    
    - Store records that fail processing for later analysis
    - Implement sideline processing for problematic records
    - Automatic or manual reprocessing capabilities
5. **Comprehensive Monitoring**
    
    - Real-time dashboards for pipeline health
    - Historical performance analysis
    - Anomaly detection for early warning

This complete data pipeline architecture provides:

- Flexibility to handle any data source or destination
- Scalability to process millions of records efficiently
- Resilience with comprehensive error handling
- Observability through detailed monitoring
- Maintainability with clean separation of concerns

I've successfully implemented similar architect
## Data Engineering & ETL (continued)

I've successfully implemented similar architectures in production environments processing 50+ million records daily, with key features including:

- Auto-scaling based on input data volume
- Geographic data partitioning for parallelization
- Hot/warm/cold data tiering for cost optimization
- Self-healing recovery from infrastructure failures
- Comprehensive data lineage tracking

This approach has consistently delivered 99.9%+ reliability while maintaining processing costs under $0.10 per million records.

## Web Development and APIs

### 8. Question: Describe how you would design a high-performance, scalable REST API using Python. Include considerations for authentication, rate limiting, caching, validation, documentation, and testing.

**Answer:**

Based on my experience building and scaling production REST APIs, I'll outline a comprehensive architecture:

**Architecture Overview:**

```
          ┌─────────────────┐
          │  Load Balancer  │
          └────────┬────────┘
                   ▼
┌─────────────────────────────────┐
│           API Gateway           │
│  (Auth, Rate Limiting, Routing) │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│        Application Servers      │
│  (FastAPI/Flask + Gunicorn)     │
└────┬───────────────────────┬────┘
     ▼                       ▼
┌────────┐              ┌────────┐
│  Cache │◄────────────►│Database│
└────────┘              └────────┘
```

Let me implement each component:

**1. API Framework Choice**

I'd use FastAPI for a modern API with excellent performance:

```python
# main.py
import logging
import time
from typing import Dict, List, Optional, Any, Union
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.openapi.docs import get_swagger_ui_html
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.database import get_db
from app.core.exceptions import APIException
from app.core.middleware import (
    RequestIdMiddleware,
    TimingMiddleware,
    RateLimitMiddleware,
    CustomLoggingMiddleware
)

# Setup logging
logger = logging.getLogger(__name__)
setup_logging()

# Create lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting API application")
    # Initialize connections, warm caches, etc.
    yield
    # Shutdown logic
    logger.info("Shutting down API application")
    # Clean up connections, etc.

# Create application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url=None,  # We'll define a custom docs endpoint
    redoc_url=None,  # We'll define a custom redoc endpoint
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestIdMiddleware)
app.add_middleware(CustomLoggingMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RateLimitMiddleware, rate_limit=settings.RATE_LIMIT)

# Exception handler
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "error_code": exc.error_code,
            "request_id": request.state.request_id
        }
    )

# Custom exception handler for 500 errors
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "An unexpected error occurred",
            "error_code": "internal_server_error",
            "request_id": request.state.request_id
        }
    )

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": settings.API_VERSION}

# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

# Include API router
app.include_router(api_router, prefix=settings.API_PREFIX)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=settings.API_WORKERS
    )
```

**2. Authentication & Authorization**

Implementing a comprehensive auth system:

```python
# app/core/auth.py
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, Union
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app.core.config import settings
from app.core.database import get_db
from app.models.user import User

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token auth
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/token")

# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: int

class TokenData(BaseModel):
    sub: str
    scopes: list[str] = []
    exp: datetime

# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)

# Token utilities
def create_access_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.JWT_SECRET_KEY, 
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Any = Depends(get_db)
) -> User:
    """Get the current authenticated user from token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode token
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        token_data = TokenData(sub=user_id, scopes=payload.get("scopes", []))
        
        # Validate expiration
        if token_data.exp < datetime.utcnow():
            raise credentials_exception
        
    except jwt.PyJWTError:
        raise credentials_exception
    
    # Get user from database
    user = await db.get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    
    return user

# Role-based authorization
def requires_scope(required_scopes: list[str]):
    """Dependency for endpoint scope authorization."""
    async def scope_validator(
        current_user: User = Depends(get_current_user)
    ) -> User:
        for scope in required_scopes:
            if scope not in current_user.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: requires scope '{scope}'",
                )
        return current_user
    
    return scope_validator

# Authentication API
async def authenticate_user(
    username: str, 
    password: str,
    db: Any
) -> Optional[User]:
    """Authenticate a user with username and password."""
    user = await db.get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
```

**3. Rate Limiting Middleware**

Implementing a Redis-based rate limiter:

```python
# app/core/middleware.py
import time
import redis
from typing import Callable, Dict, Tuple
from fastapi import Request, Response
import logging
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import settings

logger = logging.getLogger(__name__)

class RequestIdMiddleware(BaseHTTPMiddleware):
    """Add a unique request ID to each request."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response

class TimingMiddleware(BaseHTTPMiddleware):
    """Track request processing time."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

class CustomLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            }
        )
        
        response = await call_next(request)
        
        logger.info(
            f"Response: {response.status_code}",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "status_code": response.status_code,
                "process_time": response.headers.get("X-Process-Time", "unknown"),
            }
        )
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis."""
    
    def __init__(self, app: ASGIApp, rate_limit: Dict[str, int]) -> None:
        super().__init__(app)
        self.rate_limit = rate_limit  # {'requests': 100, 'period': 60}
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for certain paths
        if request.url.path in ['/health', '/metrics']:
            return await call_next(request)
        
        # Get client identifier (API key or IP address)
        client_id = self._get_client_id(request)
        
        # Check rate limit
        allowed, remaining, reset_at = self._check_rate_limit(client_id)
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {client_id}",
                extra={
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "client_id": client_id,
                }
            )
            
            return Response(
                content={"error": "Rate limit exceeded"},
                status_code=429,
                headers={
                    "X-RateLimit-Limit": str(self.rate_limit["requests"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(reset_at - int(time.time())),
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit["requests"])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get a unique identifier for the client."""
        # Try to get API key from header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key}"
        
        # Fall back to IP address
        return f"ip:{request.client.host}"
    
    def _check_rate_limit(self, client_id: str) -> Tuple[bool, int, int]:
        """Check if the request is within rate limits.
        
        Returns:
            Tuple of (allowed, remaining, reset_at)
        """
        # Get current timestamp
        now = int(time.time())
        period = self.rate_limit["period"]
        
        # Calculate time bucket (truncate to period)
        time_bucket = now - (now % period)
        reset_at = time_bucket + period
        
        # Create Redis key with time bucket for automatic expiration
        redis_key = f"ratelimit:{client_id}:{time_bucket}"
        
        # Increment counter and set expiry if needed
        requests = self.redis.incr(redis_key)
        if requests == 1:
            self.redis.expire(redis_key, period * 2)  # 2x period for safety
        
        # Check if over limit
        allowed = requests <= self.rate_limit["requests"]
        remaining = max(0, self.rate_limit["requests"] - requests)
        
        return allowed, remaining, reset_at
```

**4. Data Validation with Pydantic Models**

```python
# app/schemas/user.py
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, validator
import re

class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: bool = True

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength."""
        # Has minimum 8 characters
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        
        # Has at least one uppercase character
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase character')
        
        # Has at least one lowercase character
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase character')
        
        # Has at least one digit
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        # Has at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        return v

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    password: Optional[str] = None
    is_active: Optional[bool] = None

class UserInDBBase(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class User(UserInDBBase):
    """User model returned to clients."""
    pass

class UserInDB(UserInDBBase):
    """User model stored in database."""
    hashed_password: str
    scopes: List[str] = []
```

**5. API Endpoints with OpenAPI Documentation**

```python
# app/api/v1/endpoints/users.py
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user, requires_scope
from app.core.database import get_db
from app.models.user import User
from app.schemas.user import User as UserSchema, UserCreate, UserUpdate
from app.crud.user import user_crud

router = APIRouter()

@router.get("/", response_model=List[UserSchema])
async def get_users(
    skip: int = Query(0, ge=0, description="Skip the first N items"),
    limit: int = Query(100, ge=1, le=100, description="Limit the number of items returned"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(requires_scope(["users:read"]))
) -> Any:
    """
    Retrieve users.
    
    Requires the 'users:read' scope.
    """
    users = await user_crud.get_multi(db, skip=skip, limit=limit)
    return users

@router.post("/", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(requires_scope(["users:create"]))
) -> Any:
    """
    Create a new user.
    
    Requires the 'users:create' scope.
    """
    # Check if user already exists
    user = await user_crud.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="A user with this email already exists"
        )
    
    user = await user_crud.create(db, obj_in=user_in)
    return user

@router.get("/{user_id}", response_model=UserSchema)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(requires_scope(["users:read"]))
) -> Any:
    """
    Get a specific user by id.
    
    Requires the 'users:read' scope.
    """
    user = await user_crud.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    return user

@router.put("/{user_id}", response_model=UserSchema)
async def update_user(
    user_id: str,
    user_in: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(requires_scope(["users:write"]))
) -> Any:
    """
    Update a user.
    
    Requires the 'users:write' scope.
    """
    user = await user_crud.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    user = await user_crud.update(db, db_obj=user, obj_in=user_in)
    return user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(requires_scope(["users:delete"]))
) -> Any:
    """
    Delete a user.
    
    Requires the 'users:delete' scope.
    """
    user = await user_crud.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    await user_crud.remove(db, id=user_id)
    return None
```

**6. Caching with Redis**

```python
# app/core/cache.py
from typing import Any, Optional, TypeVar, Type, Callable, Dict, Union
import json
import logging
import inspect
import functools
import redis
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar('T')

class RedisCache:
    """Redis cache implementation."""
    
    def __init__(self, redis_url: str = settings.REDIS_URL):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = settings.CACHE_TTL
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test the Redis connection."""
        try:
            self.redis.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}")
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value from the cache."""
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache."""
        try:
            return self.redis.set(key, value, ex=ttl or self.default_ttl)
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> int:
        """Delete a value from the cache."""
        try:
            return self.redis.delete(key)
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return 0
    
    async def flush(self) -> bool:
        """Flush the cache."""
        try:
            return self.redis.flushdb()
        except Exception as e:
            logger.warning(f"Redis flush error: {e}")
            return False

# Global cache instance
cache = RedisCache()

def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    key_builder: Optional[Callable[..., str]] = None
):
    """Decorator to cache function results.
    
    Args:
        ttl: Optional time-to-live in seconds
        key_prefix: Prefix for cache keys
        key_builder: Optional function to build cache keys
    """
    def decorator(func):
        # Handle both async and sync functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    # Default key based on function name and arguments
                    arg_str = json.dumps([str(a) for a in args], sort_keys=True)
                    kwarg_str = json.dumps({k: str(v) for k, v in kwargs.items()}, sort_keys=True)
                    cache_key = f"{key_prefix}:{func.__name__}:{arg_str}:{kwarg_str}"
                
                # Try to get from cache
                cached_value = await cache.get(cache_key)
                if cached_value:
                    try:
                        # Handle different return types
                        signature = inspect.signature(func)
                        return_type = signature.return_annotation
                        
                        # If return type is a Pydantic model, parse from JSON
                        if (
                            hasattr(return_type, "__origin__") and 
                            return_type.__origin__ is Union and
                            any(issubclass(t, BaseModel) for t in return_type.__args__ if hasattr(t, "__mro__"))
                        ):
                            # Find the BaseModel in the Union
                            for t in return_type.__args__:
                                if hasattr(t, "__mro__") and issubclass(t, BaseModel):
                                    return t.parse_raw(cached_value)
                        
                        # For direct BaseModel return type
                        elif inspect.isclass(return_type) and issubclass(return_type, BaseModel):
                            return return_type.parse_raw(cached_value)
                        
                        # Otherwise, just load the JSON
                        return json.loads(cached_value)
                    
                    except Exception as e:
                        logger.warning(f"Error deserializing cached value: {e}")
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result if not None
                if result is not None:
                    try:
                        # Convert Pydantic models to JSON
                        if isinstance(result, BaseModel):
                            cached_data = result.json()
                        else:
                            cached_data = json.dumps(result)
                        
                        await cache.set(cache_key, cached_data, ttl)
                    except Exception as e:
                        logger.warning(f"Error caching result: {e}")
                
                return result
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    # Default key based on function name and arguments
                    arg_str = json.dumps([str(a) for a in args], sort_keys=True)
                    kwarg_str = json.dumps({k: str(v) for k, v in kwargs.items()}, sort_keys=True)
                    cache_key = f"{key_prefix}:{func.__name__}:{arg_str}:{kwarg_str}"
                
                # Try to get from cache
                try:
                    cached_value = cache.redis.get(cache_key)
                    if cached_value:
                        # Handle different return types
                        signature = inspect.signature(func)
                        return_type = signature.return_annotation
                        
                        # If return type is a Pydantic model, parse from JSON
                        if (
                            hasattr(return_type, "__origin__") and 
                            return_type.__origin__ is Union and
                            any(issubclass(t, BaseModel) for t in return_type.__args__ if hasattr(t, "__mro__"))
                        ):
                            # Find the BaseModel in the Union
                            for t in return_type.__args__:
                                if hasattr(t, "__mro__") and issubclass(t, BaseModel):
                                    return t.parse_raw(cached_value)
                        
                        # For direct BaseModel return type
                        elif inspect.isclass(return_type) and issubclass(return_type, BaseModel):
                            return return_type.parse_raw(cached_value)
                        
                        # Otherwise, just load the JSON
                        return json.loads(cached_value)
                except Exception as e:
                    logger.warning(f"Error getting cached value: {e}")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result if not None
                if result is not None:
                    try:
                        # Convert Pydantic models to JSON
                        if isinstance(result, BaseModel):
                            cached_data = result.json()
                        else:
                            cached_data = json.dumps(result)
                        
                        cache.redis.set(cache_key, cached_data, ex=(ttl or cache.default_ttl))
                    except Exception as e:
                        logger.warning(f"Error caching result: {e}")
                
                return result
            
            return sync_wrapper
    
    return decorator
```

**7. Comprehensive Testing Framework**

```python
# tests/conftest.py
import asyncio
import os
import pytest
import pytest_asyncio
from typing import Dict, Generator, AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.database import get_db, Base
from app.main import app
from app.models.user import User
from app.crud.user import user_crud
from app.schemas.user import UserCreate

# Override settings for testing
settings.SQLALCHEMY_DATABASE_URI = "sqlite+aiosqlite:///./test.db"
settings.TESTING = True

# Create async engine for testing
engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    poolclass=NullPool,
)
TestingSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    # Create the database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async with TestingSessionLocal() as session:
        yield session
    
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
def client(db: AsyncSession) -> Generator[TestClient, None, None]:
    """Get a TestClient instance with overridden dependencies."""
    
    async def override_get_db():
        yield db
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Remove override after test is complete
    app.dependency_overrides.clear()

@pytest_asyncio.fixture
async def test_user(db: AsyncSession) -> User:
    """Create a test user."""
    user_in = UserCreate(
        email="test@example.com",
        username="testuser",
        password="Password123!",
        full_name="Test User"
    )
    user = await user_crud.create(db, obj_in=user_in)
    return user

@pytest.fixture
def token_headers(client: TestClient, test_user
```
## Web Development and APIs (continued)

```python
@pytest.fixture
def token_headers(client: TestClient, test_user: User) -> Dict[str, str]:
    """Get authentication headers with JWT token."""
    # Get token
    login_data = {
        "username": "testuser",
        "password": "Password123!"
    }
    response = client.post(f"{settings.API_PREFIX}/auth/token", data=login_data)
    tokens = response.json()
    
    assert response.status_code == 200
    assert "access_token" in tokens
    
    return {"Authorization": f"Bearer {tokens['access_token']}"}
```

**Unit and Integration Tests**

```python
# tests/api/test_users.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User

@pytest.mark.asyncio
async def test_create_user(client: TestClient, token_headers: dict):
    data = {
        "email": "new_user@example.com",
        "username": "newuser",
        "password": "NewPassword123!",
        "full_name": "New Test User"
    }
    response = client.post(
        f"{settings.API_PREFIX}/users/",
        headers=token_headers,
        json=data
    )
    assert response.status_code == 201
    content = response.json()
    assert content["email"] == data["email"]
    assert content["username"] == data["username"]
    assert "id" in content
    assert "password" not in content

@pytest.mark.asyncio
async def test_get_existing_user(client: TestClient, token_headers: dict, test_user: User):
    response = client.get(
        f"{settings.API_PREFIX}/users/{test_user.id}",
        headers=token_headers
    )
    assert response.status_code == 200
    content = response.json()
    assert content["email"] == test_user.email
    assert content["username"] == test_user.username
    assert content["id"] == test_user.id

@pytest.mark.asyncio
async def test_get_nonexistent_user(client: TestClient, token_headers: dict):
    response = client.get(
        f"{settings.API_PREFIX}/users/nonexistent",
        headers=token_headers
    )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_update_user(client: TestClient, token_headers: dict, test_user: User):
    data = {"full_name": "Updated Name"}
    response = client.put(
        f"{settings.API_PREFIX}/users/{test_user.id}",
        headers=token_headers,
        json=data
    )
    assert response.status_code == 200
    content = response.json()
    assert content["full_name"] == data["full_name"]
    assert content["email"] == test_user.email  # Unchanged

@pytest.mark.asyncio
async def test_delete_user(client: TestClient, token_headers: dict, test_user: User):
    response = client.delete(
        f"{settings.API_PREFIX}/users/{test_user.id}",
        headers=token_headers
    )
    assert response.status_code == 204
    
    # Verify user is deleted
    response = client.get(
        f"{settings.API_PREFIX}/users/{test_user.id}",
        headers=token_headers
    )
    assert response.status_code == 404
```

**Database Schema and Models**

```python
# app/models/user.py
import uuid
from sqlalchemy import Column, String, Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    scopes = Column(ARRAY(String), default=[])
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
```

**CRUD Operations**

```python
# app/crud/base.py
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base class for CRUD operations."""
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    async def get(self, db: AsyncSession, id: Any) -> Optional[ModelType]:
        """Get a single record by ID."""
        result = await db.execute(select(self.model).filter(self.model.id == id))
        return result.scalars().first()
    
    async def get_multi(
        self, db: AsyncSession, *, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        """Get multiple records."""
        result = await db.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return result.scalars().all()
    
    async def create(self, db: AsyncSession, *, obj_in: CreateSchemaType) -> ModelType:
        """Create a new record."""
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def update(
        self,
        db: AsyncSession,
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """Update a record."""
        obj_data = jsonable_encoder(db_obj)
        
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def remove(self, db: AsyncSession, *, id: Any) -> ModelType:
        """Delete a record."""
        obj = await self.get(db, id)
        if obj:
            await db.delete(obj)
            await db.commit()
        return obj

# app/crud/user.py
from typing import Any, Dict, Optional, Union
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_password_hash
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.crud.base import CRUDBase

class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    """CRUD operations for users."""
    
    async def get_by_email(self, db: AsyncSession, *, email: str) -> Optional[User]:
        """Get a user by email."""
        result = await db.execute(select(User).filter(User.email == email))
        return result.scalars().first()
    
    async def get_by_username(self, db: AsyncSession, *, username: str) -> Optional[User]:
        """Get a user by username."""
        result = await db.execute(select(User).filter(User.username == username))
        return result.scalars().first()
    
    async def create(self, db: AsyncSession, *, obj_in: UserCreate) -> User:
        """Create a new user with hashed password."""
        db_obj = User(
            email=obj_in.email,
            username=obj_in.username,
            hashed_password=get_password_hash(obj_in.password),
            full_name=obj_in.full_name,
            is_active=obj_in.is_active
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def update(
        self,
        db: AsyncSession,
        *,
        db_obj: User,
        obj_in: Union[UserUpdate, Dict[str, Any]]
    ) -> User:
        """Update a user, handling password hashing if needed."""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        
        # Hash password if provided
        if "password" in update_data:
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password
        
        return await super().update(db, db_obj=db_obj, obj_in=update_data)

# Create instance
user_crud = CRUDUser(User)
```

**8. Deployment Configuration**

For production deployment, I'd use Docker with optimized settings:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.4.2

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy requirements
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN ~/.local/bin/poetry config virtualenvs.create false \
    && ~/.local/bin/poetry install --no-interaction --no-ansi --no-dev

# Copy application code
COPY . /app/

# Expose port
EXPOSE 8000

# Run the application with Gunicorn
CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

**Docker Compose for local development:**

```yaml
# docker-compose.yml
version: '3'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/app
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:14
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=app
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

**9. Production Scaling and Resilience**

For production, I'd use Kubernetes for orchestration:

```yaml
# kubernetes/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  labels:
    app: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: ${ECR_REPOSITORY_URI}:${IMAGE_TAG}
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        env:
        - name: ENVIRONMENT
          value: production
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: redis-url
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: jwt-secret
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      automountServiceAccountToken: false

---
apiVersion: v1
kind: Service
metadata:
  name: api
spec:
  selector:
    app: api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**10. Key Design Considerations**

1. **Performance Optimization**
    
    - Async I/O for high concurrency
    - Connection pooling for databases
    - Efficient caching of expensive operations
    - Bulk operations where appropriate
2. **Security Best Practices**
    
    - JWT with appropriate expiration and refresh mechanism
    - Password hashing with bcrypt
    - Scope-based authorization
    - Input validation with Pydantic
    - Rate limiting to prevent abuse
    - HTTPS enforcement in production
3. **Scalability**
    
    - Stateless design for horizontal scaling
    - Database connection pooling
    - Redis for distributed caching and rate limiting
    - Kubernetes for orchestration
    - Automatic scaling based on load
4. **Monitoring and Observability**
    
    - Structured logging with correlation IDs
    - Request timing metrics
    - Prometheus metrics for dashboards
    - Health check endpoints
    - Detailed error reporting
5. **Developer Experience**
    
    - Comprehensive API documentation with Swagger/ReDoc
    - Consistent error responses
    - Clear validation error messages
    - Robust testing framework

This comprehensive API design addresses all critical aspects of a production-ready system:

- **Authentication**: JWT-based with refresh tokens
- **Rate Limiting**: Redis-based with per-client tracking
- **Caching**: Multi-level with function and response caching
- **Validation**: Pydantic models with custom validators
- **Documentation**: Auto-generated OpenAPI with custom Swagger UI
- **Testing**: Pytest with async support and test fixtures

I've implemented similar architectures for APIs handling millions of requests per day, including a financial services API that required sub-100ms response times and 99.99% uptime guarantees, as well as a healthcare data API that needed to meet strict HIPAA compliance requirements.

## Distributed Systems & Microservices

### 9. Question: Design a microservices architecture for an e-commerce platform, addressing service discovery, inter-service communication, data consistency, and fault tolerance.

**Answer:**

Based on my experience building and scaling microservice architectures, I'll outline a comprehensive design for an e-commerce platform:

**Architecture Overview:**

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   API Gateway │     │ Auth Service  │     │ User Service  │     │ Product       │
│   & Routing   │     │               │     │               │     │ Catalog       │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │                     │
┌───────┴─────────────────────┴─────────────────────┴─────────────────────┴───────┐
│                                Service Mesh                                     │
└───────┬─────────────────────┬─────────────────────┬─────────────────────┬───────┘
        │                     │                     │                     │
┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
│   Cart        │     │   Order       │     │   Payment     │     │   Inventory   │
│   Service     │     │   Service     │     │   Service     │     │   Service     │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │                     │
┌───────┴─────────────────────┴─────────────────────┴─────────────────────┴───────┐
│                            Event Streaming Platform                             │
└───────┬─────────────────────┬─────────────────────┬─────────────────────┬───────┘
        │                     │                     │                     │
┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
│   Analytics   │     │   Search      │     │  Notification │     │   Shipping    │
│   Service     │     │   Service     │     │   Service     │     │   Service     │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

**1. Core Domain Services**

Let's define each service with its responsibilities:

1. **User Service**: Manages user profiles, preferences, and addresses
2. **Auth Service**: Handles authentication, authorization, and token management
3. **Product Catalog Service**: Manages product details, categories, and media
4. **Inventory Service**: Tracks stock levels, reservations, and backorders
5. **Cart Service**: Manages shopping carts and saved items
6. **Order Service**: Processes orders, orchestrates order fulfillment
7. **Payment Service**: Processes payments, refunds, and payment methods
8. **Shipping Service**: Manages shipping options, carriers, and tracking
9. **Search Service**: Provides product search and filtering capabilities
10. **Notification Service**: Sends emails, SMS, and push notifications
11. **Analytics Service**: Collects and processes user behavior and business metrics

**2. Service Discovery & API Gateway**

I'd implement service discovery using Kubernetes with a service mesh:

```yaml
# Example Kubernetes Service for Product Catalog
apiVersion: v1
kind: Service
metadata:
  name: product-service
  namespace: ecommerce
  labels:
    app: product-service
spec:
  selector:
    app: product-service
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

For the API Gateway, I'd use an Envoy-based solution like Ambassador or Istio:

```yaml
# Example API Gateway Route
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: ecommerce-gateway
spec:
  hosts:
  - "api.ecommerce.com"
  gateways:
  - ecommerce-gateway
  http:
  - match:
    - uri:
        prefix: /products
    route:
    - destination:
        host: product-service
        port:
          number: 80
  - match:
    - uri:
        prefix: /users
    route:
    - destination:
        host: user-service
        port:
          number: 80
```

**3. Inter-Service Communication**

I'd implement a combination of synchronous and asynchronous communication:

**Synchronous (REST/gRPC) for query operations:**

```python
# Python client example using gRPC for synchronous product lookup
import grpc
from product_service_pb2 import ProductRequest
from product_service_pb2_grpc import ProductServiceStub

def get_product_details(product_id: str) -> dict:
    """Get product details from Product Service using gRPC."""
    with grpc.insecure_channel('product-service:50051') as channel:
        stub = ProductServiceStub(channel)
        request = ProductRequest(product_id=product_id)
        try:
            response = stub.GetProduct(request, timeout=0.5)  # 500ms timeout
            return {
                'id': response.id,
                'name': response.name,
                'description': response.description,
                'price': response.price,
                'image_url': response.image_url
            }
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                # Handle timeout with circuit breaker pattern
                return None
            elif e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            else:
                # Log and propagate other errors
                raise
```

**Asynchronous (Event-Driven) for commands and notifications:**

```python
# Event producer for order created events
from kafka import KafkaProducer
import json
import uuid
from datetime import datetime

class OrderEventProducer:
    def __init__(self, bootstrap_servers):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda v: v.encode('utf-8')
        )
    
    def publish_order_created(self, order_data):
        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': 'order_created',
            'timestamp': datetime.utcnow().isoformat(),
            'data': order_data
        }
        
        # Send to Kafka with order_id as key for partitioning
        self.producer.send(
            'orders', 
            key=order_data['order_id'],
            value=event
        )
        
        # Ensure message is sent
        self.producer.flush()
```

**4. Data Consistency with Saga Pattern**

For transactions spanning multiple services, I'd implement the Saga pattern:

```python
# Orchestration-based saga for order processing
class OrderSagaOrchestrator:
    def __init__(self, kafka_producer, saga_store):
        self.producer = kafka_producer
        self.saga_store = saga_store  # Persistent store for saga state
    
    async def start_order_saga(self, order_data):
        """Start a new order processing saga."""
        # Generate saga ID
        saga_id = str(uuid.uuid4())
        
        # Record initial state
        saga_state = {
            'saga_id': saga_id,
            'order_id': order_data['order_id'],
            'status': 'STARTED',
            'steps': [
                {'name': 'validate_payment', 'status': 'PENDING'},
                {'name': 'reserve_inventory', 'status': 'PENDING'},
                {'name': 'create_shipment', 'status': 'PENDING'},
                {'name': 'complete_order', 'status': 'PENDING'}
            ],
            'current_step': 0,
            'compensation_required': False,
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Persist initial state
        await self.saga_store.save_saga(saga_id, saga_state)
        
        # Publish initial command to payment service
        await self.publish_validate_payment_command(saga_id, order_data)
        
        return saga_id
    
    async def handle_payment_validated(self, event):
        """Handle payment validated event."""
        saga_id = event['saga_id']
        saga = await self.saga_store.get_saga(saga_id)
        
        if not saga or saga['status'] != 'STARTED':
            return  # Ignore or log warning
        
        # Update saga state
        saga['steps'][0]['status'] = 'COMPLETED'
        saga['current_step'] = 1
        await self.saga_store.save_saga(saga_id, saga)
        
        # Publish next command
        await self.publish_reserve_inventory_command(
            saga_id, 
            event['order_id'], 
            event['payment_id']
        )
    
    async def handle_payment_failed(self, event):
        """Handle payment failed event."""
        saga_id = event['saga_id']
        saga = await self.saga_store.get_saga(saga_id)
        
        if not saga or saga['status'] != 'STARTED':
            return  # Ignore or log warning
        
        # Update saga state
        saga['steps'][0]['status'] = 'FAILED'
        saga['status'] = 'FAILED'
        saga['failure_reason'] = event.get('reason', 'Payment validation failed')
        await self.saga_store.save_saga(saga_id, saga)
        
        # Publish order failed event
        await self.publish_order_failed_event(saga_id, event['order_id'])
    
    # Similar handlers for other steps...
    
    async def publish_validate_payment_command(self, saga_id, order_data):
        """Publish command to validate payment."""
        command = {
            'command_id': str(uuid.uuid4()),
            'saga_id': saga_id,
            'command_type': 'validate_payment',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'order_id': order_data['order_id'],
                'user_id': order_data['user_id'],
                'amount': order_data['total_amount'],
                'payment_method_id': order_data['payment_method_id']
            }
        }
        
        await self.producer.send('payment-commands', key=saga_id, value=command)
    
    # Similar methods for other commands...
    
    async def compensate_saga(self, saga_id):
        """Trigger compensation for a failed saga."""
        saga = await self.saga_store.get_saga(saga_id)
        
        if not saga or saga['status'] != 'FAILED':
            return  # Nothing to compensate
        
        saga['compensation_required'] = True
        await self.saga_store.save_saga(saga_id, saga)
        
        # Determine completed steps that need compensation
        completed_steps = [
            step for step in saga['steps'] 
            if step['status'] == 'COMPLETED'
        ]
        
        # Execute compensating transactions in reverse order
        for step in reversed(completed_steps):
            if step['name'] == 'reserve_inventory':
                await self.publish_release_inventory_command(
                    saga_id, saga['order_id']
                )
            elif step['name'] == 'create_shipment':
                await self.publish_cancel_shipment_command(
                    saga_id, saga['order_id']
                )
            # Add other compensation steps...
```

**5. Fault Tolerance and Resilience**

Implementing fault tolerance patterns:

**Circuit Breaker Pattern:**

```python
import time
from functools import wraps

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    CLOSED = 'CLOSED'
    OPEN = 'OPEN'
    HALF_OPEN = 'HALF_OPEN'
    
    def __init__(
        self,
        failure_threshold=5,
        recovery_timeout=30,
        fallback_function=None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.fallback_function = fallback_function
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == self.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = self.HALF_OPEN
                else:
                    return await self._handle_open_state(*args, **kwargs)
            
            try:
                result = await func(*args, **kwargs)
                
                # Success in half-open state resets circuit breaker
                if self.state == self.HALF_OPEN:
                    self.state = self.CLOSED
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                return await self._handle_failure(e, *args, **kwargs)
        
        return wrapper
    
    async def _handle_open_state(self, *args, **kwargs):
        """Handle calls when circuit is open."""
        if self.fallback_function:
            return await self.fallback_function(*args, **kwargs)
        raise CircuitBreakerOpenException("Circuit breaker is open")
    
    async def _handle_failure(self, exception, *args, **kwargs):
        """Handle function execution failures."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if (
            self.state == self.CLOSED and 
            self.failure_count >= self.failure_threshold
        ):
            self.state = self.OPEN
        
        if self.fallback_function:
            return await self.fallback_function(*args, **kwargs)
        
        raise exception

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass

# Example usage with inventory service client
async def fallback_get_product_stock(product_id):
    """Fallback function when inventory service is down."""
    # Return a conservative default
    return {"product_id": product_id, "in_stock": False, "quantity": 0}

@CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60,
    fallback_function=fallback_get_product_stock
)
async def get_product_stock(product_id):
    """Get product stock information from Inventory Service."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://inventory-service/api/stock/{product_id}",
            timeout=0.5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
```

**Rate Limiting and Bulkheads:**

```python
import asyncio
import time
from functools import wraps

class RateLimiter:
    """Rate limiter implementation."""
    
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.perio
```

## Distributed Systems & Microservices (continued)

```python
class RateLimiter:
    """Rate limiter implementation."""
    
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove expired timestamps
            self.calls = [t for t in self.calls if now - t < self.period]
            
            # Check if we've reached the limit
            if len(self.calls) >= self.max_calls:
                raise RateLimitExceededException(
                    f"Rate limit of {self.max_calls} calls per {self.period}s exceeded"
                )
            
            # Record the call
            self.calls.append(now)
            
            # Execute the function
            return await func(*args, **kwargs)
        
        return wrapper

class Bulkhead:
    """Bulkhead pattern implementation."""
    
    def __init__(self, max_concurrent_calls):
        self.max_concurrent_calls = max_concurrent_calls
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)
    
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.semaphore:
                return await func(*args, **kwargs)
        
        return wrapper

class RateLimitExceededException(Exception):
    """Exception raised when rate limit is exceeded."""
    pass

# Example usage with inventory service
@RateLimiter(max_calls=50, period=1.0)  # 50 requests per second
@Bulkhead(max_concurrent_calls=20)      # Maximum 20 concurrent calls
async def update_product_inventory(product_id, delta):
    """Update product inventory in Inventory Service."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://inventory-service/api/inventory/update",
            json={"product_id": product_id, "quantity_delta": delta},
            timeout=1.0
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
```

**6. Event-Driven Architecture for Eventual Consistency**

Implementing event sourcing and CQRS for the Order service:

```python
# Order aggregate with event sourcing
class OrderAggregate:
    """Order aggregate root with event sourcing."""
    
    def __init__(self, order_id):
        self.order_id = order_id
        self.status = None
        self.items = []
        self.shipping_address = None
        self.payment_method = None
        self.total_amount = 0
        self.created_at = None
        self.events = []  # Stores uncommitted events
    
    @classmethod
    async def load(cls, order_id, event_store):
        """Load an order from event store."""
        order = cls(order_id)
        events = await event_store.get_events(order_id)
        
        for event in events:
            order.apply_event(event, is_new=False)
        
        return order
    
    def apply_event(self, event, is_new=True):
        """Apply an event to the aggregate."""
        event_type = event['event_type']
        data = event['data']
        
        if event_type == 'OrderCreated':
            self.status = 'CREATED'
            self.items = data['items']
            self.shipping_address = data['shipping_address']
            self.payment_method = data['payment_method']
            self.total_amount = data['total_amount']
            self.created_at = data['created_at']
        
        elif event_type == 'OrderPaymentProcessed':
            self.status = 'PAYMENT_PROCESSED'
        
        elif event_type == 'OrderItemsReserved':
            self.status = 'ITEMS_RESERVED'
        
        elif event_type == 'OrderShipmentCreated':
            self.status = 'SHIPMENT_CREATED'
            self.shipping_tracking = data['tracking_info']
        
        elif event_type == 'OrderCompleted':
            self.status = 'COMPLETED'
        
        elif event_type == 'OrderCancelled':
            self.status = 'CANCELLED'
            self.cancellation_reason = data.get('reason')
        
        # Store new events for later persistence
        if is_new:
            self.events.append(event)
    
    def create_order(self, user_id, items, shipping_address, payment_method):
        """Create a new order."""
        if self.status is not None:
            raise ValueError("Order already exists")
        
        # Calculate total amount
        total_amount = sum(item['price'] * item['quantity'] for item in items)
        
        event = {
            'event_type': 'OrderCreated',
            'data': {
                'order_id': self.order_id,
                'user_id': user_id,
                'items': items,
                'shipping_address': shipping_address,
                'payment_method': payment_method,
                'total_amount': total_amount,
                'created_at': datetime.utcnow().isoformat()
            }
        }
        
        self.apply_event(event)
        return self
    
    def process_payment(self, payment_id, transaction_id):
        """Mark payment as processed."""
        if self.status != 'CREATED':
            raise ValueError(f"Cannot process payment for order in {self.status} state")
        
        event = {
            'event_type': 'OrderPaymentProcessed',
            'data': {
                'order_id': self.order_id,
                'payment_id': payment_id,
                'transaction_id': transaction_id,
                'processed_at': datetime.utcnow().isoformat()
            }
        }
        
        self.apply_event(event)
        return self
    
    def reserve_items(self, inventory_reservation_id):
        """Mark items as reserved in inventory."""
        if self.status != 'PAYMENT_PROCESSED':
            raise ValueError(f"Cannot reserve items for order in {self.status} state")
        
        event = {
            'event_type': 'OrderItemsReserved',
            'data': {
                'order_id': self.order_id,
                'inventory_reservation_id': inventory_reservation_id,
                'reserved_at': datetime.utcnow().isoformat()
            }
        }
        
        self.apply_event(event)
        return self
    
    # Additional methods for other state transitions...
    
    async def save(self, event_store, event_publisher=None):
        """Save uncommitted events to the event store."""
        if not self.events:
            return
        
        # Save to event store
        await event_store.save_events(self.order_id, self.events)
        
        # Publish events
        if event_publisher:
            for event in self.events:
                await event_publisher.publish_event(
                    topic='order-events',
                    event_type=event['event_type'],
                    key=self.order_id,
                    data=event['data']
                )
        
        # Clear uncommitted events
        self.events = []

# Order service with CQRS - Command side
class OrderCommandService:
    """Command side of the Order service."""
    
    def __init__(self, event_store, event_publisher):
        self.event_store = event_store
        self.event_publisher = event_publisher
    
    async def create_order(self, order_data):
        """Create a new order."""
        order_id = str(uuid.uuid4())
        
        # Create and persist order
        order = OrderAggregate(order_id).create_order(
            user_id=order_data['user_id'],
            items=order_data['items'],
            shipping_address=order_data['shipping_address'],
            payment_method=order_data['payment_method']
        )
        
        await order.save(self.event_store, self.event_publisher)
        
        return order_id
    
    async def process_payment(self, order_id, payment_data):
        """Process payment for an order."""
        # Load order
        order = await OrderAggregate.load(order_id, self.event_store)
        
        # Update and persist
        order.process_payment(
            payment_id=payment_data['payment_id'],
            transaction_id=payment_data['transaction_id']
        )
        
        await order.save(self.event_store, self.event_publisher)
    
    async def reserve_inventory(self, order_id, reservation_id):
        """Reserve inventory for an order."""
        order = await OrderAggregate.load(order_id, self.event_store)
        order.reserve_items(reservation_id)
        await order.save(self.event_store, self.event_publisher)
    
    # Additional command handlers...

# Order service with CQRS - Query side
class OrderQueryService:
    """Query side of the Order service."""
    
    def __init__(self, order_repository):
        self.repository = order_repository
    
    async def get_order(self, order_id):
        """Get order details by ID."""
        return await self.repository.find_by_id(order_id)
    
    async def get_user_orders(self, user_id, status=None, limit=10, offset=0):
        """Get orders for a specific user."""
        filters = {"user_id": user_id}
        if status:
            filters["status"] = status
        
        return await self.repository.find(filters, limit=limit, offset=offset)
    
    async def search_orders(self, query, limit=10, offset=0):
        """Search orders by various criteria."""
        return await self.repository.search(query, limit=limit, offset=offset)

# Event consumer to update read models
class OrderEventConsumer:
    """Consumes order events to update read models."""
    
    def __init__(self, order_repository, kafka_consumer):
        self.repository = order_repository
        self.consumer = kafka_consumer
    
    async def start(self):
        """Start consuming events."""
        await self.consumer.subscribe(['order-events'])
        await self.process_events()
    
    async def process_events(self):
        """Process events from Kafka."""
        try:
            while True:
                message = await self.consumer.getone()
                
                event_type = message.headers.get('event_type').decode('utf-8')
                event_data = json.loads(message.value.decode('utf-8'))
                
                await self.handle_event(event_type, event_data)
                
                # Commit offset
                await self.consumer.commit()
                
        except Exception as e:
            logger.error(f"Error processing events: {e}")
            raise
    
    async def handle_event(self, event_type, event_data):
        """Handle a specific event type."""
        order_id = event_data['order_id']
        
        if event_type == 'OrderCreated':
            # Create new order in read model
            await self.repository.create({
                'order_id': order_id,
                'user_id': event_data['user_id'],
                'items': event_data['items'],
                'shipping_address': event_data['shipping_address'],
                'payment_method': event_data['payment_method'],
                'total_amount': event_data['total_amount'],
                'status': 'CREATED',
                'created_at': event_data['created_at'],
                'updated_at': datetime.utcnow().isoformat()
            })
        
        elif event_type in ('OrderPaymentProcessed', 'OrderItemsReserved',
                           'OrderShipmentCreated', 'OrderCompleted', 'OrderCancelled'):
            # Update existing order
            status_map = {
                'OrderPaymentProcessed': 'PAYMENT_PROCESSED',
                'OrderItemsReserved': 'ITEMS_RESERVED',
                'OrderShipmentCreated': 'SHIPMENT_CREATED',
                'OrderCompleted': 'COMPLETED',
                'OrderCancelled': 'CANCELLED'
            }
            
            update_data = {
                'status': status_map[event_type],
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Add event-specific fields
            if event_type == 'OrderShipmentCreated':
                update_data['shipping_tracking'] = event_data['tracking_info']
            elif event_type == 'OrderCancelled':
                update_data['cancellation_reason'] = event_data.get('reason')
            
            await self.repository.update(order_id, update_data)
```

**7. Service Mesh and Configuration**

I'd implement a service mesh like Istio for advanced traffic management:

```yaml
# Traffic management with Istio for canary deployments
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: order-service
spec:
  hosts:
  - order-service
  http:
  - route:
    - destination:
        host: order-service
        subset: v1
      weight: 90
    - destination:
        host: order-service
        subset: v2
      weight: 10
---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: order-service
spec:
  host: order-service
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 10
        maxRequestsPerConnection: 10
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
```

**8. Monitoring and Observability**

Implementing distributed tracing:

```python
# Tracing with OpenTelemetry
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Set up tracer
def setup_tracing(app, service_name):
    trace.set_tracer_provider(TracerProvider())
    jaeger_exporter = JaegerSpanExporter(
        service_name=service_name,
        agent_host_name=os.getenv("JAEGER_HOST", "jaeger"),
        agent_port=int(os.getenv("JAEGER_PORT", "6831")),
    )
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

# Using the tracer in code
tracer = trace.get_tracer(__name__)

@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    with tracer.start_as_current_span("get_order") as span:
        # Add attributes to span
        span.set_attribute("order_id", order_id)
        
        try:
            order = await order_service.get_order(order_id)
            if not order:
                span.set_attribute("error", True)
                span.set_attribute("error.message", "Order not found")
                return {"error": "Order not found"}, 404
            
            span.set_attribute("order.status", order["status"])
            return order
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", True)
            raise
```

**9. Deployment and Scaling**

Kubernetes manifest for one of the services:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cart-service
  namespace: ecommerce
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cart-service
  template:
    metadata:
      labels:
        app: cart-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
    spec:
      containers:
      - name: cart-service
        image: ecommerce/cart-service:v1.2.3
        ports:
        - containerPort: 8080
        env:
        - name: SERVICE_NAME
          value: "cart-service"
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: redis-host
        - name: KAFKA_BOOTSTRAP_SERVERS
          valueFrom:
            configMapKeyRef:
              name: kafka-config
              key: bootstrap-servers
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          limits:
            cpu: 500m
            memory: 512Mi
          requests:
            cpu: 100m
            memory: 128Mi
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 20
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cart-service
  namespace: ecommerce
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cart-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**10. Security Considerations**

Implementing OAuth2/OIDC for authentication:

```python
# OAuth2 authentication middleware
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
import jwt
from jwt.exceptions import InvalidTokenError

# Set up OAuth2 with authorization code flow
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{settings.AUTH_SERVER_URL}/authorize",
    tokenUrl=f"{settings.AUTH_SERVER_URL}/token"
)

async def get_current_user_id(token: str = Depends(oauth2_scheme)):
    """Validate access token and return user ID."""
    try:
        # Verify token with Auth0 or other OIDC provider's public key
        payload = jwt.decode(
            token,
            settings.JWT_PUBLIC_KEY,
            algorithms=["RS256"],
            audience=settings.API_AUDIENCE,
            issuer=settings.AUTH_SERVER_URL
        )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user_id
    
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Checking permissions from JWT claims
def has_permission(required_permission: str):
    """Check if user has required permission."""
    async def _has_permission(token: str = Depends(oauth2_scheme)):
        try:
            payload = jwt.decode(
                token,
                settings.JWT_PUBLIC_KEY,
                algorithms=["RS256"],
                audience=settings.API_AUDIENCE,
                issuer=settings.AUTH_SERVER_URL
            )
            
            # Extract permissions from payload
            permissions = payload.get("permissions", [])
            
            if required_permission not in permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {required_permission} required"
                )
            
            return payload
            
        except InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    return _has_permission

# Using the permissions check in an endpoint
@app.post("/orders")
async def create_order(
    order: OrderCreate,
    user_claims=Depends(has_permission("create:orders"))
):
    # User ID from the token
    user_id = user_claims["sub"]
    
    # Create order
    order_id = await order_service.create_order(user_id, order)
    
    return {"order_id": order_id}
```

**Key Design Considerations:**

1. **Service Boundaries**
    
    - Defined around business capabilities
    - Isolated data stores for each service
    - Clear ownership and responsibility
2. **Communication Patterns**
    
    - Synchronous communication for queries (REST/gRPC)
    - Asynchronous communication for commands (Kafka)
    - Event-driven architecture for loose coupling
3. **Data Consistency**
    
    - Saga pattern for distributed transactions
    - Event sourcing for audit and recovery
    - CQRS to optimize for different use cases
4. **Resilience**
    
    - Circuit breakers to prevent cascading failures
    - Bulkheads to isolate failures
    - Rate limiting to protect resources
    - Retry with exponential backoff
5. **Scalability**
    
    - Stateless services for horizontal scaling
    - Kubernetes for orchestration
    - Auto-scaling based on metrics
    - Caching for performance
6. **Observability**
    
    - Distributed tracing with OpenTelemetry
    - Centralized logging with ELK stack
    - Metrics and dashboards with Prometheus/Grafana
    - Health checks and alerting
7. **Security**
    
    - Service-to-service authentication
    - Fine-grained authorization
    - Secure communication (mTLS)
    - Secrets management

This architecture provides a solid foundation for an e-commerce platform that can scale to millions of users while maintaining resilience and consistency. The event-driven approach enables eventual consistency for complex operations, while the service mesh provides advanced traffic management and security features.

I've implemented similar architectures for large-scale e-commerce platforms, including one that handled over 5 million daily active users and processed thousands of orders per minute during peak sales events. The key challenges were maintaining inventory accuracy during flash sales and ensuring order fulfillment even when certain services experienced temporary outages.

## System Design & Scalability

### 10. Question: Design a real-time analytics system that can ingest and process millions of events per minute, with sub-second query response times for dashboards and alerts.

**Answer:**

Based on my experience building high-throughput analytics systems, I'll design a comprehensive solution for real-time analytics that can handle millions of events per minute while providing sub-second query responses.

**Architecture Overview:**

```
Data Sources → Ingestion Layer → Processing Layer → Storage Layer → Query Layer → Presentation Layer
     ↑              ↑                 ↑                 ↑              ↑              ↑
     └──────────────┴─────────────────┴─────────────────┴──────────────┴──────────────┘
                                          ↓
                               Monitoring & Management Layer
```

Let's analyze each component:

## 1. Ingestion Layer

The ingestion layer needs to handle high-throughput data collection without any data loss:

```python
# Event collection API using FastAPI and Kafka
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json
from aiokafka import AIOKafkaProducer
import logging

app = FastAPI(title="Analytics Ingestion API")
logger = logging.getLogger("ingestion-api")

# Configure Kafka producer
producer = None

@app.on_event("startup")
async def startup_event():
    global producer
    producer = AIOKafkaProducer(
        bootstrap_servers='kafka-1:9092,kafka-2:9092,kafka-3:9092',
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
        compression_type="snappy",
        acks="all",
        max_batch_size=16384,
        linger_ms=5,  # Wait 5ms to collect batches
        max_request_size=1048576,
        buffer_memory=67108864  # 64MB buffer
    )
    await producer.start()

@app.on_event("shutdown")
async def shutdown_event():
    if producer:
        await producer.stop()

class Event(BaseModel):
    event_type: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    source: str
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    properties: Dict[str, Any] = {}

class EventBatch(BaseModel):
    events: List[Event]
    batch_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

@app.post("/events", status_code=202)
async def ingest_events(event_batch: EventBatch, background_tasks: BackgroundTasks):
    """Ingest a batch of events."""
    background_tasks.add_task(process_event_batch, event_batch)
    return {"batch_id": event_batch.batch_id, "event_count": len(event_batch.events)}

async def process_event_batch(event_batch: EventBatch):
    """Process a batch of events asynchronously."""
    try:
        events = event_batch.dict()["events"]
        
        # Enrich and validate events
        for event in events:
            # Add server timestamp if missing
            if not event.get("timestamp"):
                event["timestamp"] = datetime.utcnow().isoformat()
            
            # Add correlation ID for tracing
            event["correlation_id"] = event_batch.batch_id
        
        # Send events to Kafka topics based on event_type
        futures = []
        for event in events:
            # Get topic based on event type
            event_type = event["event_type"]
            topic = f"events-{event_type.lower().replace('_', '-')}"
            
            # Send to Kafka with key for partitioning
            key = event.get("user_id") or event.get("session_id") or event.get("client_id") or "default"
            future = await producer.send_and_wait(
                topic, 
                value=event,
                key=key.encode('utf-8') if key else None
            )
            futures.append(future)
        
        # Ensure all messages are sent
        await asyncio.gather(*futures)
        
    except Exception as e:
        logger.error(f"Error processing event batch {event_batch.batch_id}: {str(e)}")
        # In production, we'd send to a dead-letter queue for retry
```

For high-volume data collection, we'd also implement a direct SDK for clients:

```python
# Python client SDK for event collection
import requests
import json
import uuid
import time
import threading
import queue
from datetime import datetime
from typing import Dict, Any, List, Optional

class AnalyticsClient:
    """Client SDK for sending events to the analytics service."""
    
    def __init__(
        self, 
        api_url: str, 
        api_key: str, 
        client_id: str = None,
        batch_size: int = 25, 
        flush_interval: float = 1.0,
        max_queue_size: int = 10000
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.client_id = client_id or str(uuid.uuid4())
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Event queue for batching
        self.queue = queue.Queue(maxsize=max_queue_size)
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.user_id = None
        
        # Start background thread for flushing
        self.running = True
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
    
    def track(self, event_type: str, properties: Dict[str, Any] = None):
        """Track an event."""
        event = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "python-sdk",
            "client_id": self.client_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "properties": properties or {}
        }
        
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            # If queue is full, drop the event and log warning
            print(f"Warning: Event queue full, dropping event {event_type}")
    
    def identify(self, user_id: str, traits: Dict[str, Any] = None):
        """Identify a user."""
        self.user_id = user_id
        
        # Send identify event
        self.track("identify", traits or {})
    
    def flush(self):
        """Manually flush the event queue."""
        events = []
        try:
            while not self.queue.empty() and len(events) < self.batch_size:
                events.append(self.queue.get_nowait())
                self.queue.task_done()
        except queue.Empty:
            pass
        
        if events:
            self._send_batch(events)
    
    def shutdown(self):
        """Shutdown the client and flush any pending events."""
        self.running = False
        self.flush_thread.join(timeout=self.flush_interval * 2)
        self.flush()  # Final flush
    
    def _flush_worker(self):
        """Worker thread to periodically flush events."""
        while self.running:
            time.sleep(self.flush_interval)
            
            try:
                # Collect events from queue
                events = []
                while not self.queue.empty() and len(events) < self.batch_size:
                    events.append(self.queue.get_nowait())
                    self.queue.task_done()
                
                if events:
                    self._send_batch(events)
            except Exception as e:
                print(f"Error in flush worker: {str(e)}")
    
    def _send_batch(self, events: List[Dict[str, Any]]):
        """Send a batch of events to the API."""
        if not events:
            return
        
        batch = {
            "events": events,
            "batch_id": str(uuid.uuid4())
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/events",
                json=batch,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Client-ID": self.client_id
                },
                timeout=5.0
            )
            
            if response.status_code != 202:
                print(f"Error sending events: {response.status_code} {response.text}")
                # In production, we'd implement retry logic here
        
        except Exception as e:
            print(f"Error sending events: {str(e)}")
```