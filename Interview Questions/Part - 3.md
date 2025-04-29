
# 14. Find Nth Highest Salary - SQL Query (continued)

**Approach 4: Using Self-Join (Older SQL versions):**

```sql
SELECT DISTINCT e1.salary
FROM employees e1
WHERE N-1 = (
    SELECT COUNT(DISTINCT e2.salary)
    FROM employees e2
    WHERE e2.salary > e1.salary
);
```

**Example:** For a table `employees` with the following data:

```
| id | name  | salary |
|----|-------|--------|
| 1  | Alice | 100000 |
| 2  | Bob   | 80000  |
| 3  | Carol | 90000  |
| 4  | Dave  | 90000  |
| 5  | Eve   | 110000 |
```

Finding the 2nd highest salary:

```sql
-- Using DENSE_RANK
SELECT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as rank
    FROM employees
) ranked_salaries
WHERE rank = 2;
```

Result: 100000 (Alice's salary)

Note: DENSE_RANK() treats duplicate values as the same rank, while RANK() would skip ranks.

# 15. When to Choose SQL vs NoSQL

**When to Choose SQL:**

1. **Data Structure Consistency**:
    
    - When you have structured data with a consistent schema
    - When relationships between data are well-defined and important
2. **ACID Transactions**:
    
    - When you need strict data integrity and transaction support
    - For financial systems, payment processing, booking systems
3. **Complex Queries**:
    
    - When you need to perform complex queries with JOINs
    - For reporting, analytics, and business intelligence
4. **Established Schema**:
    
    - When your data schema is relatively stable and doesn't change often
    - For legacy systems and applications with predictable data models
5. **Regulatory Compliance**:
    
    - When you need to ensure data consistency for regulatory requirements
    - For systems that require auditing and strict data governance

**When to Choose NoSQL:**

1. **Scalability Requirements**:
    
    - When horizontal scaling is needed for massive data or traffic
    - For applications that need to handle huge user loads
2. **Schema Flexibility**:
    
    - When your data structure evolves frequently or is unpredictable
    - For rapid development environments where schema might change
3. **High Write Throughput**:
    
    - When you need to handle massive write operations
    - For logging, IoT data, time-series data, real-time analytics
4. **Specialized Data Models**:
    
    - Document databases (MongoDB) for hierarchical, document-like data
    - Graph databases (Neo4j) for highly connected data with complex relationships
    - Key-value stores (Redis) for caching and simple data structures
    - Wide-column stores (Cassandra) for time-series and event logging
5. **Geographical Distribution**:
    
    - When data needs to be distributed across multiple regions
    - For global applications requiring low-latency data access worldwide

**Decision Matrix:**

|Factor|SQL|NoSQL|
|---|---|---|
|Data Structure|Structured, relational|Varied (structured/unstructured)|
|Schema|Fixed|Flexible, schema-less|
|Scaling|Vertical (primarily)|Horizontal|
|Transactions|ACID compliant|Varies (BASE in many cases)|
|Query Complexity|Complex joins|Typically simpler queries|
|Data Volume|Medium to large|Very large to massive|
|Data Consistency|Strong consistency|Often eventual consistency|
|Development Speed|Slower for changes|Faster for iterative changes|
|Community & Support|Mature, extensive|Growing, varies by database|
|Use Case Examples|Financial systems, CRM|Social media, big data, IoT|

# 16. Kafka vs RabbitMQ

|Feature|Apache Kafka|RabbitMQ|
|---|---|---|
|**Core Model**|Distributed log/stream processing platform|Traditional message broker|
|**Messaging Pattern**|Publish-subscribe with log-based approach|Support for various patterns (P2P, pub-sub, RPC)|
|**Message Retention**|Retains messages for configurable time/size|Removes messages once consumed (by default)|
|**Performance**|Very high throughput (millions/sec)|Moderate throughput (tens of thousands/sec)|
|**Ordering**|Strict ordering within partitions|Best-effort ordering (can be guaranteed with queues)|
|**Scalability**|Horizontally scalable, partition-based|Clustered, with primary-replica architecture|
|**Consumer Model**|Pull-based (consumers request messages)|Push-based (broker pushes to consumers)|
|**Use Cases**|Log aggregation, stream processing, event sourcing|Traditional queuing, work distribution, RPC|
|**Complexity**|More complex to set up and maintain|Easier to set up and use|
|**Message Routing**|Simple topic-based routing|Advanced routing with exchanges and bindings|
|**Protocol Support**|Custom binary protocol over TCP|AMQP, MQTT, STOMP, HTTP|
|**Message Priority**|Not supported natively|Supported|
|**Dead Letter Queue**|Not built-in (can be achieved with patterns)|Built-in support|
|**Message Size**|Better for small to medium messages|Handles large messages better|

**When to Choose Kafka:**

1. **High Throughput Requirements**:
    
    - When processing millions of messages per second
    - For log aggregation, metrics collection
2. **Event Sourcing / Streaming**:
    
    - When you need to store and process events in order
    - For building event-driven architectures
3. **Data Retention**:
    
    - When you need to replay messages or access message history
    - For auditing, reprocessing historical data
4. **Multiple Consumers**:
    
    - When multiple systems need to process the same data independently
    - For building complex data pipelines
5. **Long-term Message Storage**:
    
    - When messages are part of your business data

**When to Choose RabbitMQ:**

1. **Complex Routing**:
    
    - When sophisticated message routing is needed
    - For systems with complex message distribution patterns
2. **Protocol Flexibility**:
    
    - When multiple messaging protocols are required
    - For bridging different systems using different standards
3. **Traditional Messaging Patterns**:
    
    - When you need request-reply, RPC patterns
    - For task distribution in worker queues
4. **Message Reliability**:
    
    - When guaranteed delivery is critical
    - For financial transactions, critical business processes
5. **Lower Latency**:
    
    - When millisecond-level processing is important
    - For real-time user-facing applications

# 17. How to Containerize Your Source Code

**Basic Containerization Steps:**

1. **Create a Dockerfile**:
    
    - Define the base image
    - Set up environment
    - Copy source code
    - Install dependencies
    - Configure entry point
2. **Build the Docker Image**:
    
    - Run `docker build` command
    - Tag the image appropriately
3. **Run and Test the Container**:
    
    - Verify the application works as expected
    - Check for any environment-specific issues
4. **Push to a Container Registry**:
    
    - Share the image with others
    - Make it available for deployment

**Example Dockerfile for a Python Application:**

```dockerfile
# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
```

**Example Dockerfile for a Node.js Application:**

```dockerfile
# Base image
FROM node:16-alpine

# Set working directory
WORKDIR /usr/src/app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application files
COPY . .

# Expose the port
EXPOSE 3000

# Run the application
CMD ["node", "server.js"]
```

**Best Practices:**

1. **Use Official Base Images**:
    
    - Start with official, minimal images
    - Use specific version tags (not `latest`)
2. **Multi-Stage Builds**:
    
    - Separate build and runtime environments
    - Reduces final image size
3. **Layer Optimization**:
    
    - Order commands to leverage caching
    - Combine related commands using `&&`
4. **Security Considerations**:
    
    - Run as non-root user
    - Scan images for vulnerabilities
    - Remove unnecessary tools and packages
5. **Configuration**:
    
    - Use environment variables for configuration
    - Consider using config files for complex setups
6. **Volumes**:
    
    - Use volumes for persistent data
    - Separate application and data concerns

**Example of a Multi-Stage Build (Java):**

```dockerfile
# Build stage
FROM maven:3.8-openjdk-11 AS build
WORKDIR /app
COPY pom.xml .
# Download dependencies first (better caching)
RUN mvn dependency:go-offline
COPY src/ ./src/
RUN mvn package -DskipTests

# Runtime stage
FROM openjdk:11-jre-slim
WORKDIR /app
COPY --from=build /app/target/*.jar app.jar
EXPOSE 8080
# Create a non-root user
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser
USER appuser
ENTRYPOINT ["java", "-jar", "app.jar"]
```

**Docker Commands for Containerization:**

```bash
# Build an image
docker build -t myapp:1.0 .

# Run a container
docker run -p 8080:8080 myapp:1.0

# Push to registry
docker tag myapp:1.0 registry.example.com/myapp:1.0
docker push registry.example.com/myapp:1.0
```

# 18. Why We Need Helm Charts and Standard Commands

**Why We Need Helm:**

1. **Package Management**:
    
    - Bundles Kubernetes resources into a single package (chart)
    - Makes applications shareable and reusable
2. **Configuration Management**:
    
    - Parameterizes Kubernetes manifests via templates
    - Supports different environments with value overrides
3. **Release Management**:
    
    - Tracks installed releases and their versions
    - Supports upgrades, rollbacks, and versioning
4. **Complex Deployments**:
    
    - Manages dependencies between components
    - Handles ordering and synchronization of resources
5. **Consistency and Standardization**:
    
    - Provides a standard way to define applications
    - Creates repeatable deployments

**Standard Helm Commands:**

1. **Chart Management:**

```bash
# Create a new chart
helm create mychart

# Package a chart
helm package mychart

# Lint a chart to find issues
helm lint mychart

# Add a chart repository
helm repo add stable https://charts.helm.sh/stable

# Update chart repositories
helm repo update
```

2. **Installation and Deployment:**

```bash
# Install a chart
helm install my-release mychart

# Install with custom values
helm install my-release mychart -f values-custom.yaml

# Install with set values
helm install my-release mychart --set key1=value1,key2=value2

# Install from repository
helm install my-release stable/nginx-ingress
```

3. **Release Management:**

```bash
# List releases
helm list

# Get release status
helm status my-release

# Get release history
helm history my-release

# Upgrade a release
helm upgrade my-release mychart --set version=2.0.0

# Rollback a release
helm rollback my-release 1

# Uninstall a release
helm uninstall my-release
```

4. **Chart Inspection:**

```bash
# Show chart information
helm show chart mychart

# Show chart values
helm show values mychart

# See generated templates
helm template my-release mychart
```

5. **Plugins and Additional Features:**

```bash
# List installed plugins
helm plugin list

# Add a plugin
helm plugin install URL

# Pull chart from repository
helm pull stable/mysql
```

**Helm Chart Structure:**

```
mychart/
├── Chart.yaml          # Metadata about the chart
├── values.yaml         # Default configuration values
├── charts/             # Directory of dependencies
├── templates/          # Directory of template files
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   └── _helpers.tpl    # Template helpers
└── templates/NOTES.txt # Usage notes displayed after install
```

# 19. How API Versioning Happens

**Common API Versioning Strategies:**

1. **URI Path Versioning**:
    
    ```
    https://api.example.com/v1/resources
    https://api.example.com/v2/resources
    ```
    
    - Pros: Explicit, easy to understand, works with all clients
    - Cons: Not RESTful (resources should be independent of version)
2. **Query Parameter Versioning**:
    
    ```
    https://api.example.com/resources?version=1
    https://api.example.com/resources?version=2
    ```
    
    - Pros: Doesn't change resource URI, backward compatible
    - Cons: Easy to miss, might be lost in request forwarding
3. **Header Versioning**:
    
    ```
    GET /resources HTTP/1.1
    Accept-Version: v1
    
    GET /resources HTTP/1.1
    Accept-Version: v2
    ```
    
    - Pros: Keeps URLs clean, follows HTTP protocol
    - Cons: Less visible, harder to test in browser
4. **Accept Header Versioning (Content Negotiation)**:
    
    ```
    GET /resources HTTP/1.1
    Accept: application/vnd.example.v1+json
    
    GET /resources HTTP/1.1
    Accept: application/vnd.example.v2+json
    ```
    
    - Pros: Most RESTful approach, follows HTTP spec
    - Cons: Complex, harder to debug, less client support
5. **Subdomain Versioning**:
    
    ```
    https://v1.api.example.com/resources
    https://v2.api.example.com/resources
    ```
    
    - Pros: Cleanly separates major versions, allows separate deployments
    - Cons: Requires additional DNS configuration, can be problematic for cookies

**Implementation Considerations:**

1. **When to Create a New Version**:
    
    - Breaking changes to request/response format
    - Removal of resources or properties
    - Significant changes in business logic
    - Changes in resource relationships
2. **Versioning Granularity**:
    
    - API-level versioning: Entire API gets a version
    - Resource-level versioning: Individual resources get versions
    - Field-level versioning: Specific fields get versions
3. **Version Lifecycle Management**:
    
    - Communicate deprecation schedules clearly
    - Maintain older versions for a reasonable time
    - Consider offering migration tools
4. **Backward Compatibility**:
    
    - Maintain compatibility where possible
    - Add new fields instead of modifying existing ones
    - Use optional parameters instead of required ones

**Example Implementation (Spring Boot):**

```java
// URI Path Versioning
@RestController
@RequestMapping("/v1/users")
public class UserControllerV1 {
    @GetMapping("/{id}")
    public UserDtoV1 getUserV1(@PathVariable String id) {
        // Implementation for V1
    }
}

@RestController
@RequestMapping("/v2/users")
public class UserControllerV2 {
    @GetMapping("/{id}")
    public UserDtoV2 getUserV2(@PathVariable String id) {
        // Implementation for V2
    }
}

// Header Versioning
@RestController
@RequestMapping("/users")
public class UserController {
    @GetMapping(value = "/{id}", headers = "X-API-Version=1")
    public UserDtoV1 getUserV1(@PathVariable String id) {
        // Implementation for V1
    }
    
    @GetMapping(value = "/{id}", headers = "X-API-Version=2")
    public UserDtoV2 getUserV2(@PathVariable String id) {
        // Implementation for V2
    }
}
```

# 20. Why REST or gRPC and When to Use Which

**REST (Representational State Transfer):**

**Advantages:**

1. **Simplicity and Familiarity**:
    
    - Uses standard HTTP methods and status codes
    - Easily understood by most developers
2. **Client Compatibility**:
    
    - Works with virtually any client that speaks HTTP
    - No special client libraries needed
3. **Statelessness**:
    
    - Each request contains all information needed
    - Easier to scale horizontally
4. **Caching**:
    
    - Uses HTTP's built-in caching mechanisms
    - Improves performance for read-heavy operations
5. **Tooling Ecosystem**:
    
    - Wide range of tools for development, testing, monitoring
    - Documentation systems like Swagger/OpenAPI

**gRPC (Google Remote Procedure Call):**

**Advantages:**

1. **Performance**:
    
    - Uses HTTP/2 for multiplexing, streaming
    - Protocol Buffers for efficient serialization
2. **Strong Typing**:
    
    - Contract-first approach with Protocol Buffers
    - Code generation for multiple languages
3. **Bi-directional Streaming**:
    
    - Support for unary, server, client, and bi-directional streaming
    - Real-time communication capabilities
4. **Built-in Features**:
    
    - Timeouts, cancellation, authentication
    - Load balancing, service discovery integration
5. **Multi-language Support**:
    
    - Automatic client/server code generation
    - Consistent behavior across languages

**When to Use REST:**

1. **Public APIs**:
    
    - When external developers need to consume your API
    - When clients may be unknown or varied
2. **Browser Clients**:
    
    - When JavaScript in browsers needs to call your API
    - For web applications using AJAX
3. **Simple CRUD Operations**:
    
    - For basic Create, Read, Update, Delete operations
    - When resource-oriented design fits naturally
4. **Hypermedia and Resource Discovery**:
    
    - When clients need to discover API capabilities
    - For highly interconnected resources (HATEOAS)
5. **When HTTP Benefits Matter**:
    
    - When caching is important
    - When leveraging HTTP ecosystem (proxies, CDNs)

**When to Use gRPC:**

1. **Microservices Architecture**:
    
    - For internal service-to-service communication
    - When service contracts need to be strictly defined
2. **Performance-Critical Systems**:
    
    - When latency matters
    - For high-throughput systems
3. **Real-time Communication**:
    
    - When bidirectional streaming is needed
    - For systems with push notifications or updates
4. **Polyglot Environments**:
    
    - When services are written in different languages
    - For consistent cross-language APIs
5. **Limited Network Bandwidth**:
    
    - When message size and efficiency matter
    - For mobile applications or IOT devices

**Comparison Table:**

|Aspect|REST|gRPC|
|---|---|---|
|**Protocol**|HTTP 1.1 (typically)|HTTP/2|
|**Format**|JSON, XML, etc. (text-based)|Protocol Buffers (binary)|
|**Design Style**|Resource-oriented|Action-oriented (RPC)|
|**Contract**|OpenAPI/Swagger (optional)|Protocol Buffers (required)|
|**Code Generation**|Optional, varies by tool|Native, consistent across languages|
|**Streaming**|Limited (no native bidirectional)|Fully supported (all types)|
|**Browser Support**|Native|Requires proxy (gRPC-Web)|
|**Learning Curve**|Low|Medium|
|**Payload Size**|Larger|Smaller (3-10x more compact)|
|**Latency**|Higher|Lower|
|**Use Cases**|Public APIs, web clients|Microservices, performance-critical|

# 21. Kubernetes Architecture and Core Components

**High-Level Architecture:**

Kubernetes follows a master-worker architecture:

1. **Control Plane (Master)**: Manages the cluster
2. **Node (Worker)**: Runs workloads (containers)

**Control Plane Components:**

1. **API Server**:
    
    - Front-end for Kubernetes control plane
    - Exposes the Kubernetes API
    - Processes RESTful requests and updates etcd
2. **etcd**:
    
    - Distributed key-value store
    - Stores all cluster data and state
    - Provides reliable, consistent data storage
3. **Scheduler**:
    
    - Watches for new Pods without assigned nodes
    - Selects optimal nodes for Pods based on constraints
    - Considers resources, affinity, taints/tolerations
4. **Controller Manager**:
    
    - Runs controller processes
    - Regulates cluster state
    - Includes Node Controller, Replication Controller, Endpoints Controller, etc.
5. **Cloud Controller Manager**:
    
    - Interfaces with cloud provider APIs
    - Manages cloud-specific resources (load balancers, storage, etc.)
    - Allows cloud-specific and core code to evolve independently

**Node Components:**

1. **Kubelet**:
    
    - Agent running on each node
    - Ensures containers are running in Pods
    - Communicates with API server and reports node health
2. **Kube-proxy**:
    
    - Network proxy on each node
    - Implements the Kubernetes Service concept
    - Handles pod networking and load balancing
3. **Container Runtime**:
    
    - Software for running containers (Docker, containerd, CRI-O)
    - Responsible for container lifecycle management
    - Pulls images and runs containers

**Addons:**

1. **DNS Server (CoreDNS)**:
    
    - Service discovery within cluster
    - Resolves service names to IP addresses
2. **Dashboard**:
    
    - Web-based UI for cluster management
    - Visualizes cluster resources
3. **Network Plugin**:
    
    - Implements Container Network Interface (CNI)
    - Examples: Calico, Flannel, Cilium, Weave Net
4. **Metrics Server**:
    
    - Collects resource metrics
    - Used for autoscaling and monitoring

**Communication Flow:**

1. **User Interaction**:
    
    - User applies manifest via kubectl
    - kubectl communicates with API server
2. **API Server Processing**:
    
    - API server validates and processes request
    - Stores resource definitions in etcd
    - Notifies relevant controllers
3. **Controller Actions**:
    
    - Controllers watch for changes
    - Take actions to achieve desired state
    - Update resource status in etcd
4. **Scheduler Placement**:
    
    - Scheduler assigns Pods to nodes
    - Node assignment stored in etcd
5. **Kubelet Execution**:
    
    - Kubelet watches for assigned Pods
    - Instructs container runtime to run containers
    - Reports status back to API server

**Visualization:**

```
                    ┌─────────────────────────────────────────┐
                    │           Control Plane                 │
                    │                                         │
┌──────────┐        │  ┌──────────┐        ┌──────────────┐   │
│          │        │  │          │        │              │   │
│ kubectl  │◄─────────►│ API      │◄─────► │ etcd         │   │
│          │        │  │ Server   │        │              │   │
└──────────┘        │  └─────┬────┘        └──────────────┘   │
                    │        │                                 │
                    │  ┌─────▼────┐        ┌───────────────┐  │
                    │  │          │        │               │  │
                    │  │Scheduler │        │ Controller    │  │
                    │  │          │        │ Manager       │  │
                    │  └──────────┘        └───────────────┘  │
                    └─────────────────────────────────────────┘
                              │                 │
                              ▼                 ▼
            ┌─────────────────────────┐ ┌─────────────────────────┐
            │        Node 1           │ │        Node 2           │
            │                         │ │                         │
            │  ┌─────────┐ ┌────────┐ │ │  ┌─────────┐ ┌────────┐ │
            │  │         │ │        │ │ │  │         │ │        │ │
            │  │ Kubelet │ │ Kube-  │ │ │  │ Kubelet │ │ Kube-  │ │
            │  │         │ │ proxy  │ │ │  │         │ │ proxy  │ │
            │  └─────────┘ └────────┘ │ │  └─────────┘ └────────┘ │
            │        │                │ │        │                │
            │  ┌─────▼──────────────┐ │ │  ┌─────▼──────────────┐ │
            │  │                    │ │ │  │                    │ │
            │  │ Container Runtime  │ │ │  │ Container Runtime  │ │
            │  │                    │ │ │  │                    │ │
            │  └────────────────────┘ │ │  └────────────────────┘ │
            │                         │ │                         │
            │  ┌────────┐ ┌────────┐  │ │  ┌────────┐ ┌────────┐  │
            │  │        │ │        │  │ │  │        │ │        │  │
            │  │ Pod A  │ │ Pod B  │  │ │  │ Pod C  │ │ Pod D  │  │
            │  │        │ │        │  │ │  │        │ │        │  │
            │  └────────┘ └────────┘  │ │  └────────┘ └────────┘  │
            └─────────────────────────┘ └─────────────────────────┘
```

# 22. Standard Kubernetes Resources

1. **Pods**:
    
    - Smallest deployable unit
    - One or more containers with shared storage/network
    - When to use: For single containers or tightly coupled container groups
2. **ReplicaSet**:
    
    - Ensures a specified number of Pod replicas are running
    - Provides self-healing through automatic replacements
    - When to use: When direct control over replica count is needed
3. **Deployment**:
    
    - Manages ReplicaSets and provides declarative updates
    - Supports rolling updates and rollbacks
    - When to use: For stateless applications (default for most workloads)
4. **StatefulSet**:
    
    - Manages stateful applications with stable identities
    - Ordered deployment, scaling, and updates
    - When to use: For databases, distributed systems requiring stable network identities
5. **DaemonSet**:
    
    - Ensures a copy of a Pod runs on all (or selected) nodes
    - Automatically adds Pods to new nodes
    - When to use: For node-level operations (monitoring, logging, storage)
6. **Job**:
    
    - Runs Pods to completion
    - Tracks successful completions
    - When to use: For batch processing, one-time computations
7. **CronJob**:
    
    - Creates Jobs on a schedule
    - Cron-like syntax for scheduling
    - When to use: For scheduled jobs (backups, reports, cleanups)
8. **Service**:
    
    - Provides stable networking for Pods
    - Types: ClusterIP, NodePort, LoadBalancer, ExternalName
    - When to use: To expose applications within or outside the cluster
9. **Ingress**:
    
    - Manages external access to Services
    - HTTP/HTTPS routing, SSL termination, name-based virtual hosting
    - When to use: For HTTP-based routing to multiple services
10. **ConfigMap**:
    
    - Stores non-sensitive configuration data
    - Can be consumed as environment variables, command-line args, or files
    - When to use: For application configuration
11. **Secret**:
    
    - Stores sensitive data (passwords, tokens, keys)
    - Base64 encoded (not encrypted by default)
    - When to use: For sensitive configuration and credentials
12. **PersistentVolume (PV)**:
    
    - Represents physical storage
    - Lifecycle independent of Pods
    - When to use: For underlying storage resources
13. **PersistentVolumeClaim (PVC)**:
    
    - Request for storage by users
    - Claims a PersistentVolume
    - When to use: For applications requesting storage
14. **StorageClass**:
    
    - Defines storage provisioner and parameters
    - Enables dynamic provisioning
    - When to use: To define different classes of storage
15. **NetworkPolicy**:
    
    - Specifies how groups of Pods can communicate
    - Pod-level network firewall
    - When to use: To secure communication between Pods
16. **ServiceAccount**:
    
    - Identity for processes in Pods
    - Used for authentication and authorization
    - When to use: When Pods need to interact with the API server
17. **Role and RoleBinding**:
    
    - Defines permissions within a namespace
    - Maps users/groups to Roles
    - When to use: For namespace-scoped access control
18. **ClusterRole and ClusterRoleBinding**:
    
    - Defines permissions across the cluster
    - Maps users/groups to ClusterRoles
    - When to use: For cluster-wide access control
19. **HorizontalPodAutoscaler (HPA)**:
    
    - Automatically scales Pods based on metrics
    - Supports CPU, memory, and custom metrics
    - When to use: For automatic scaling based on load
20. **VerticalPodAutoscaler (VPA)**:
    
    - Recommends and sets resource requests
    - Adjusts CPU and memory allocations
    - When to use: For optimizing resource allocation

**Resource Selection Decision Tree:**

```
Start
 │
 ├─ Need to run containers?
 │   ├─ Yes, just once → Job
 │   ├─ Yes, on a schedule → CronJob
 │   ├─ Yes, on every node → DaemonSet
 │   ├─ Yes, with stable identity/storage → StatefulSet
 │   └─ Yes, stateless application → Deployment
 │
 ├─ Need networking?
 │   ├─ Internal access → Service (ClusterIP)
 │   ├─ External access on specific ports → Service (NodePort)
 │   ├─ External access with cloud LB → Service (LoadBalancer)
 │   └─ HTTP/HTTPS routing → Ingress
 │
 ├─ Need configuration?
 │   ├─ Non-sensitive → ConfigMap
 │   └─ Sensitive → Secret
 │
 ├─ Need persistent storage?
 │   ├─ Storage definition → PersistentVolume
 │   ├─ Storage request → PersistentVolumeClaim
 │   └─ Storage classes → StorageClass
 │
 └─ Need access control?
     ├─ Namespace scoped → Role/RoleBinding
     └─ Cluster scoped → ClusterRole/ClusterRoleBinding
```

