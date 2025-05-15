# Interview Preparation Checklist

## 1. Experience at BetterPlace

### Sample Answer:
At BetterPlace, I've worked as a SDE-3 Backend Developer for nearly 6 years. I was responsible for designing and implementing backend services across various domains, including microservices architecture, Excel processing, notification systems, candidate onboarding, and payroll processing.

I worked heavily with Python (Django) and Node.js, leveraging tools like Kafka, RabbitMQ, Redis, Elasticsearch, and deploying on Kubernetes clusters via AWS/GCP. A key part of my role was to write scalable APIs, optimize data pipelines, and provide mentorship to junior developers.

### Real-Time Examples:

**Example 1 – Microservices Architecture**

I led the design of a new microservice for Bank Account Verification and Payouts. This service communicated with external banking APIs, handled retry mechanisms using RabbitMQ, and stored transactional logs in PostgreSQL and Elasticsearch. It scaled to support thousands of daily verifications.

**Example 2 – Excel Upload/Download System**

We had a use case where HR teams uploaded Excel sheets with candidate details. I built an async processing system using Celery and Redis to validate and store data. On download, large reports were fetched from Elasticsearch, converted to Excel, and stored in S3 for download—all built using Python and integrated into our platform UI.

## 2. How did you build the Excel processing system?

### Sample Answer:
We needed a way to allow teams to upload and download large Excel files—often with 50,000+ rows. I designed an asynchronous processing system using:
- FastAPI as an endpoint to upload Excel
- Celery workers with Redis to process and validate in background
- Openpyxl for reading and writing Excel files
- Data stored in PostgreSQL or Elasticsearch depending on use case

To support downloads, I added a system where report filters were stored, Elasticsearch was queried, and the results were paginated and written to an Excel file using streaming mode (write-only mode). The file was uploaded to S3, and a real time notification sent to the user via Redis pub method.

### Real-Time Examples:

**Example 1 – Bulk Candidate Upload**

An operations team would upload Excel sheets with 10k+ candidates. We validated each row asynchronously, checked for duplicate phone numbers or email IDs, and logged errors in a downloadable report. This drastically reduced manual entry errors.

**Example 2 – Custom Report Download**

A compliance team needed filtered data (e.g., candidates from a region in last 30 days). We queried Elasticsearch using date filters, converted the data into Excel using a streaming writer (to prevent memory bloat), uploaded to S3, and gave a secure download link.

## 3. Tell me about a time you handled notification delivery at scale

### Sample Answer:
We needed to send OTPs for candidate onboarding. Initially, we used direct API calls to SMS vendors, but we faced issues with retries and throughput.

I built a notification pipeline using:
- SQS for queueing notifications with the priority based if it is OTP then this will be in high priority and it message is promotional then sent the sms in out of the office time. 
- RabbitMQ for retry handling
- Worker pool in Python to consume messages and call GupShup api
- Redis used for rate-limiting and deduplication
- Failure notifications were logged in Elasticsearch for visibility

This system scaled to handle 100k+ OTPs/day with robust retry and fallback mechanisms.

### Real-Time Examples:

**Example 1 – High Volume OTP During Bulk Onboarding**

During festival hiring drives, 10k+ candidates were onboarded in 2 days. Our notification system handled spikes using autoscaled workers on Kubernetes, ensuring real-time delivery and retries without downtime.

**Example 2 – Delayed OTP Investigation**

We noticed OTP delays for some users. I used logs from Elasticsearch and found specific SQS messages were stuck due to oversized payloads. We fixed it by trimming message content and implemented content validation at producer level.

## 4. What is the difference between Kafka and RabbitMQ, and where did you use each?

### Sample Answer:
RabbitMQ is a message broker following the traditional queue-based model, supporting complex routing and reliability. It's great for task queues.

Kafka, on the other hand, is a distributed event-streaming platform optimized for high-throughput, ordered, and persistent logs. It's better suited for event-driven architecture and analytics pipelines.

### Real-Time Examples:

**Example 1 – Using RabbitMQ for Task Queues**

We used RabbitMQ for onboarding-related tasks like document verification, email notification, and webhook callbacks. It allowed us to retry failed tasks and route them based on priority queues.

**Example 2 – Using Kafka for Event Streaming**

For our payroll pipeline, we streamed attendance events into Kafka. Multiple consumers read this stream—some for analytics, others for payroll computation. Kafka helped us decouple the event producers from the consumers.

## 5. How did you integrate Exotel and what was the benefit?

### Sample Answer:
Exotel is a cloud telephony provider. We used it to confirm candidate onboarding by initiating automated IVR calls that asked the candidate to press 1 to confirm.

I integrated Exotel using:
- Exotel APIs for call initiation
- Redis to ensure each candidate got only one call
- Async webhook handlers to process response (e.g., user pressed 1 or call failed)
- Logged response in PostgreSQL and exposed analytics via dashboards

This automation reduced the need for manual follow-up calls.

### Real-Time Examples:

**Example 1 – Reducing TAT for Onboarding**

Before integration, ops teams made 3–4 manual calls per candidate. Post Exotel integration, 70% of confirmations were automated, cutting TAT by half and saving hundreds of man-hours.

**Example 2 – Dynamic Language Selection**

We enhanced the system to select IVR language based on candidate's region. This improved success rate of confirmations, especially in non-English speaking areas.

## 6. How did you use Elasticsearch in your projects?

### Sample Answer:
I used Elasticsearch mainly for **search and analytics** purposes. We stored large volumes of candidate, attendance, and payroll data to enable fast search, filtering, and reporting.  

My responsibilities included:
- Designing index mappings based on expected queries
- Creating aggregations for reporting
- Optimizing queries with pagination, sorting, and filtering
- Managing data lifecycle using ILM (Index Lifecycle Management)

For example, report downloads would query Elasticsearch instead of hitting PostgreSQL to reduce latency and offload transactional DBs.

### Real-Time Examples:

**Example 1 – Candidate Report Filtering**

An HR team needed to filter candidates based on multiple fields like location, status, and onboarding date. We used a bool query with nested filters in Elasticsearch and reduced query time from 5s (PostgreSQL) to under 200ms.

**Example 2 – Payroll Analytics**

We generated monthly salary analytics by aggregating attendance events. Elasticsearch's terms and date_histogram aggregations allowed us to build dashboards that showed real-time stats with minimal load.

## 7. How do you mentor junior developers?

### Sample Answer:
I actively mentored 2–3 junior developers during their onboarding and project development phase. My mentorship included:
- Conducting code reviews and giving structured feedback
- Explaining system architecture and design principles
- Helping them debug complex issues instead of just giving solutions
- Assigning progressively harder tasks based on their growth

I believe in enabling them to think critically, not just finish JIRA tickets.

### Real-Time Examples:

**Example 1 – Code Review & Refactor**

A junior dev submitted a tightly coupled module. I walked them through separation of concerns, added test cases, and they later refactored the entire module using best practices. Their confidence improved, and the codebase became more maintainable.

**Example 2 – Live Debugging Session**

A team member was stuck on a production bug involving race conditions in RabbitMQ consumers. I organized a live debugging session using logs and thread profiling. We solved the issue and documented it in the internal wiki for future reference.

## 8. What cloud services have you used (AWS & GCP)?

### Sample Answer:
I have worked with both **AWS and GCP**, depending on the client or internal infrastructure.

In **AWS**, I have used:
- **EKS** (Elastic Kubernetes Service) for running microservices
- **S3** for storing Excel reports and documents
- **SQS** for queueing OTP notifications
- **Lambda** for lightweight background tasks (like sending alerts)

In **GCP**, I used:
- **GKE** for container orchestration
- **Cloud Storage** similar to S3
- **BigQuery** for running large-scale analytics in one internal project

### Real-Time Examples:

**Example 1 – Excel Report with AWS Lambda**

After generating Excel reports, we used an AWS Lambda to auto-upload it to S3 and send download links via email. This decoupled the notification logic from the main worker and improved response times.

**Example 2 – Cloud-Native Payroll Processing**

On GCP, I helped migrate our attendance processing system to run on GKE. We used GCP Pub/Sub (similar to Kafka) for event flow and Cloud Storage to archive historical logs.

## 9. How do you design scalable APIs?

### Sample Answer:
To design scalable APIs, I follow these principles:
- **Statelessness** to ensure horizontal scalability
- **Pagination and filtering** to control payload size
- **Asynchronous tasks** for long-running jobs
- **Caching** at various levels (Redis, CDN)
- **Proper logging & monitoring** to identify bottlenecks

I also ensure **backward compatibility**, using versioning and feature flags where needed.

### Real-Time Examples:

**Example 1 – Download API**

We implemented a download API that triggered a background job to generate a report, uploaded it to S3, and sent a notification. This made the endpoint responsive and scalable, even with 50k+ rows.

**Example 2 – Candidate Search API**

We built an API to search across millions of candidates. Elasticsearch queries were pre-filtered with user roles, and response size was limited via pagination. Redis cached the frequent queries, reducing response time to 150ms on average.

## 10. Can you design a scalable payroll system?

### Sample Answer:
Yes, here's how I'd approach it:

**Components:**
- **Attendance Service** – Records in/out events (Kafka events)
- **Leave Management** – Tracks leaves and approvals
- **Payroll Engine** – Consumes attendance + leave + salary rules
- **Calculation Worker** – Uses Celery or background workers
- **Report Generator** – Generates payslips (PDF/Excel)

Each module can be containerized and scaled independently using Kubernetes.

### Real-Time Examples:

**Example 1 – Real-Time Attendance Stream**

We streamed biometric attendance logs to Kafka. The payroll engine consumed it, processed data in hourly batches, and updated salaries in near real time.

**Example 2 – Excel Payslip Automation**

Once payroll was processed, payslips were auto-generated as Excel and stored in S3. HR could download individual or bulk payslips. This replaced a manual 2-day process with a 1-click automation.

## 11. Design a system for processing large Excel file uploads asynchronously

### Interview Intent:
To test your understanding of asynchronous job handling, data validation, fault tolerance, and system reliability.  

### High-Level Design:

**Components:**
- **Frontend**: Upload UI with progress indicator
- **Backend API**: Accepts the Excel file and enqueues a task
- **Queue**: Redis/RabbitMQ/Kafka
- **Worker Service**:
  - Validates data (e.g., phone, email)
  - Stores valid rows in DB
  - Logs errors and stores failed rows
- **Notification Service**: Alerts user on success/failure
- **Storage**: S3 for file storage, PostgreSQL for data

**Tech Stack:**
- FastAPI + Celery + Redis/RabbitMQ
- PostgreSQL + Elasticsearch for search
- S3 for file storage

**Key Features:**
- Async processing for performance
- Retry mechanism for resilience
- Downloadable error report

### Real-Time Examples:

**Example 1 – Candidate Bulk Upload**

Ops team uploaded 50k candidates for a client. File went through a validation pipeline, invalid rows were rejected with detailed reasons, and a downloadable error sheet was generated. Success rate improved by 30% due to early validation.

**Example 2 – Excel Row Streaming with OpenPyXL**

To prevent memory issues, we used openpyxl in **write-only mode** and processed the Excel file **row by row**, avoiding full-file loading. We later migrated this service to AWS Lambda for better cost control and autoscaling.

## 12. Design a scalable OTP Notification System

### High-Level Design:

**Components:**
- **Frontend**: Triggers OTP via REST API
- **API Gateway**: Auth and rate limit
- **Notification Producer Service**: Validates and publishes messages to a queue (SQS/RabbitMQ)
- **Notification Consumer**: Listens to queue, calls vendor API (Exotel, Twilio)
- **Failure Handling**: Retry queues or DLQs (Dead Letter Queues)
- **Monitoring & Logs**: Elasticsearch, Prometheus, Grafana

**Scalability Features:**
- Horizontal scaling of consumers
- Vendor failover (e.g., Exotel to Twilio fallback)
- Throttling per phone number (using Redis)

### Real-Time Examples:

**Example 1 – SQS + RabbitMQ Hybrid Model**

You implemented SQS for durability and RabbitMQ for retry logic. During peak loads (50k+ OTP/day), messages were never lost, and retries were handled efficiently without manual intervention.

**Example 2 – Failure Recovery**

When Exotel failed for a specific telecom circle, your team switched to Twilio using a feature flag. Logs were indexed in Elasticsearch, and customer service could quickly investigate by mobile number.

## 13. Design a microservice for bank account verification and payouts

### High-Level Design:

**Modules:**
- **Auth Service**: Handles OAuth2/JWT
- **Bank Verification Service**:
  - API to submit bank details
  - Calls 3rd-party (e.g., Razorpay/ICICI API)
  - Stores response and audit trail
- **Payout Service**:
  - Queued payouts (async)
  - Status polling / webhook handling
- **Notification Service**: Sends confirmation messages
- **Monitoring & Retry Logic**: Track failures and auto-retries

**Security:**
- Mask account numbers in logs
- Tokenized storage or Vault integration

### Real-Time Examples:

**Example 1 – Razorpay Integration**

You integrated Razorpay's bank account verification. Added rate-limited background polling and webhook fallback. 98% of verifications completed within 30 seconds.

**Example 2 – Delayed Payouts Debugging**

Used Kafka to track payout events, and a sidecar service to reattempt stuck transactions every 5 minutes. Alerting was built via Prometheus + Slack.

## 14. Design a Payroll System for Contract Workers

### Core Requirements:
- Daily/hourly wage computation
- Attendance and leave input
- Automated salary generation
- Excel/PDF payslip generation
- Manual overrides + approval flow

**Design Approach:**
- **Microservices**:
  - AttendanceService (Kafka consumer)
  - LeaveService
  - PayrollEngine (logic + rules)
  - PayslipService (Excel + PDF generator)
- **Event-Driven Flow**: Kafka streams attendance data → triggers payroll calc
- **Storage**: PostgreSQL (relational logic), Elasticsearch (reporting)

### Real-Time Examples:

**Example 1 – Incremental Payroll Calculation**

You ran partial payroll processing every night for all "active" contract workers. This helped ops verify salary in real time and catch anomalies before end-of-month.

**Example 2 – Bulk Payslip Generation**

You batched payslip generation using Celery workers. Each worker used openpyxl for Excel, and then converted it to PDF using libreoffice in headless mode. Thousands of payslips were generated in 30 minutes.

## 15. Design a Role-Based Access Control System (RBAC) for Microservices

### Requirements:
- Fine-grained access per API/module
- Tenant isolation (multi-org)
- Permissions dynamically updatable
- Auditing of changes

**System Design:**
- **UserService**: Stores user/org roles
- **AuthMiddleware**: Injects user role in request
- **PermissionService**:
  - Fetches endpoint-level permissions
  - Cached in Redis for low latency
- **AuditService**: Logs access changes, permission grants

### Real-Time Examples:

**Example 1 – Staffing App Access**

Your system enforced permissions like "can_view_salary_data" or "can_edit_attendance" via JWT claims + Redis cache. HR users had edit access, while ops had read-only access.

**Example 2 – Dynamic Role Assignment**

You exposed a UI to manage roles. Backend used FastAPI + PostgreSQL. Updates were event-driven and reflected in Redis within seconds, without requiring logout/login.

## 16. Design a Notification Service (for OTPs, emails, and voice calls)

### Problem:
Build a system that can handle multiple types of notifications with retry logic, priority handling, and vendor fallback.

### Components:
- NotificationManager: Accepts user input (email, phone, message)
- Queue: RabbitMQ or Kafka
- NotificationDispatcher: Picks the right channel (SMS, voice, email)
- VendorAdapters: Abstraction for Twilio, Exotel, etc.
- RetryScheduler: Handles failed sends with exponential backoff
- NotificationStore: Logs sent/delivery/failure states

### Class Structure (Python-like pseudocode):

```python
class Notification:
    def __init__(self, recipient, message, type):
        self.recipient = recipient
        self.message = message
        self.type = type  # sms, email, voice

class VendorAdapter:
    def send(self, notification): pass

class ExotelAdapter(VendorAdapter):
    def send(self, notification):
        # Call Exotel API

class Dispatcher:
    def dispatch(self, notification: Notification):
        if notification.type == 'sms':
            ExotelAdapter().send(notification)
        elif notification.type == 'email':
            SMTPAdapter().send(notification)
```

## 17. Design an Excel Processing Engine

### Problem:
Given structured Excel files, validate, process, and store records asynchronously.

### Class Design:

```python
class ExcelProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def process(self):
        for row in self.read_rows():
            result = Validator.validate(row)
            if result.valid:
                DB.insert(row)
            else:
                ErrorLogger.log(row, result.errors)

class Validator:
    @staticmethod
    def validate(row):
        # check email, phone, required fields
        return ValidationResult(valid=True)
```

### Trade-Offs:
- **Streaming (row-by-row)** avoids memory bloat but may limit validation rules across rows.
- **Async processing** decouples user from UI wait but requires robust monitoring.
- **Schema flexibility** is easier with JSON + schema registry than with Excel files.

## 18. Redis vs PostgreSQL for Temporary Storage

### Trade-Offs:

| **Aspect** | **Redis** | **PostgreSQL** |
|------------|-----------|----------------|
| Speed | In-memory, lightning fast | Slower (disk-based) |
| Persistence | Optional | Always persistent |
| Use Case | OTPs, rate-limits, caching | Core data, relational joins |
| Cost | Higher memory cost | Disk is cheaper |
| Durability | Needs backup config | Reliable out-of-the-box |

### Real-Time Scenario:
- ✅ Use Redis to cache user session tokens and OTPs with expiry.
- ✅ Use PostgreSQL to store verified user data, audit logs, and Excel data.

## 19. Kafka vs RabbitMQ

| **Feature** | **Kafka** | **RabbitMQ** |
|-------------|-----------|--------------|
| Message Retention | Persistent log (can re-read) | Queue-like, message deleted on ack |
| Throughput | Very high | Medium |
| Order Guarantee | Per-partition | Per-queue |
| Consumer Model | Pull-based | Push-based |
| Use Case | Event streaming, analytics | Task queues, retries |

### Real-Time Scenario:
- ✅ You used **Kafka** to stream attendance events for analytics.
- ✅ You used **RabbitMQ** to handle retryable, one-time tasks like document generation and webhook triggers.

## 20. Excel vs JSON for Bulk Data Exchange

| **Feature** | **Excel** | **JSON** |
|-------------|-----------|----------|
| Human-Readable | Yes (non-technical users) | Less friendly |
| Structured Format | Tabular (columns/rows) | Nested |
| Validation | Difficult to enforce schema | Easy with JSON Schema or Pydantic |
| Size Handling | Harder with large files | Easier with streaming parsers |

### Real-Time Scenario:
- ✅ You used **Excel** for manual HR uploads/downloads.
- ✅ You considered **JSON** for internal microservice communication due to validation and structure benefits.

## 21. Design a Role-Based Access Control (RBAC) in Code

### Problem:
Support permissions like can_edit_salary, can_view_candidate with org-level isolation.

### Class Design:

```python
class Role:
    def __init__(self, name, permissions):
        self.name = name
        self.permissions = permissions  # list of strings

class User:
    def __init__(self, id, roles):
        self.id = id
        self.roles = roles

    def has_permission(self, permission):
        return any(permission in role.permissions for role in self.roles)

# Usage
if user.has_permission("can_download_excel"):
    allow()
```

### Trade-Offs:
- **Token-based roles** are fast but hard to revoke dynamically.
- **DB-checked roles** are slower but up-to-date.
- You can use Redis to cache role-permission mappings per user/org.

## 22. Mock Behavioral Interview Questions (with answers)

These questions test how you handle teamwork, pressure, leadership, and failures.

### Q1: Tell me about a time you handled a production incident

**Answer:**

At BetterPlace, we once had a production issue where OTP delivery dropped suddenly. Users couldn't log in, and support tickets spiked.

**What I did:**
- Checked message queues and found a sudden spike in failed sends.
- Dug into logs (stored in Elasticsearch) and traced the issue to an expired Exotel API token.
- I rotated the token manually, requeued failed messages, and added a health-check alert for token expiry.

**Result:**
- We restored service within 30 minutes.
- I documented the fix and implemented a scheduled pre-expiry alert system using Prometheus.

**Key Takeaway:** Quick triage, root cause analysis, and building long-term preventive solutions.

### Q2: Tell me about a project you're most proud of

**Answer:**

I'm proud of building the **Excel processing engine** at BetterPlace.

We had manual onboarding data entered in Excel. HR uploaded large files (up to 100K rows), and the old system would crash or timeout.

I designed an async system using **Celery + Redis** that validated and stored rows in parallel. It also generated downloadable error reports and success notifications.

**Result:**
- Upload time dropped from 30 minutes to ~2 minutes async
- We improved error visibility and reduced support tickets

**Key Takeaway:** Solving user pain points with clean, scalable architecture.

### Q3: How do you handle disagreement with a teammate?

**Answer:**

During the design of a payout service, a teammate suggested embedding business logic inside the controller. I preferred separating concerns using a service layer.

Instead of pushing back immediately, I proposed we review a similar module with a layered architecture. I showed how testability and scalability improved in that design.

We ended up following the service-layer pattern for future modules as well.

**Key Takeaway:** Empathy + evidence > ego. Focus on maintainability and scalability.

## 23. Design Trade-Off Deep Dive: Async vs Sync

| **Factor** | **Async Job (e.g., Celery)** | **Sync Request (e.g., HTTP API)** |
|------------|-------------------------------|-----------------------------------|
| Response Time | Immediate (fire-and-forget) | User waits for result |
| Scalability | Highly scalable | Limited by app server threads |
| Reliability | Retry, failure tracking | If it fails, user has to retry manually |
| Monitoring | Requires job tracking dashboard | Easier with HTTP status codes |
| User Experience | Better for long tasks (Excel, Payout) | Better for quick tasks (OTP, login) |

**Conclusion:** Use **async** for long-running or bulk operations (e.g., Excel uploads, payslip generation), and **sync** for fast transactional APIs (login, profile update).

## 24. Design Trade-Off Deep Dive: Microservices vs Monolith

| **Aspect** | **Microservices** | **Monolith** |
|------------|-------------------|--------------|
| Codebase | Modular, separate deployments | Single codebase |
| Deployment | Independently deployable | All-at-once |
| Scaling | Scale individual parts | Scale whole app |
| Operational Overhead | High (infra, observability) | Lower (simple setup) |
| Use Case at BetterPlace | You split Authorization, Excel, Payouts | Earlier monolith onboarding service |
