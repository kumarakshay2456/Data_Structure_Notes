
# FastAPI Interview Question Bank (100+ Q&A with Code)

> **How to use this**: Skim the sections you’re weakest in, then practice by running the snippets. Most code blocks are copy‑paste ready.  

---

## Table of Contents
1. Basics (1–10)  
2. Routing & Path Operations (11–20)  
3. Parameters & Requests (21–30)  
4. Validation & Pydantic (31–40)  
5. Responses & Serialization (41–50)  
6. Dependencies & DI (51–60)  
7. Auth & Security (61–72)  
8. Middleware & Lifespan (73–78)  
9. Background Tasks & Scheduling (79–83)  
10. Error Handling (84–89)  
11. Async & Concurrency (90–95)  
12. OpenAPI & Docs (96–100)  
13. Testing (101–108)  
14. DB, ORMs & Transactions (109–118)  
15. Performance, Config & Deployment (119–130)  
16. WebSockets & Streaming (131–136)  
17. Project Structure & Patterns (137–144)  
18. Integrations & Misc (145–152)

---

## 1) Basics (1–10)

**Q1. What is FastAPI and why is it “fast”?**  
**A.** A modern, ASGI‑based Python web framework using type hints + Pydantic for validation and Starlette/Uvicorn for speed (async I/O).

```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/ping")
def ping():
    return {"ok": True}
```

**Q2. How does FastAPI differ from Flask/Django?**  
**A.** Built‑in type‑driven validation, async‑first, automatic OpenAPI docs; Flask is micro WSGI, Django is batteries‑included MVC.

**Q3. What is ASGI and why does it matter?**  
**A.** Async Server Gateway Interface—enables non‑blocking concurrency (websockets, long‑polling) and better throughput.

**Q4. What provides auto docs in FastAPI?**  
**A.** OpenAPI schema + interactive docs via Swagger UI (/docs) and ReDoc (/redoc).

**Q5. What’s Pydantic’s role?**  
**A.** Data parsing/validation and typed models for request/response.

```python
from pydantic import BaseModel

class UserIn(BaseModel):
    username: str
    age: int
```

**Q6. How are routes defined?**  
**A.** Path operation decorators: `@app.get`, `@app.post`, etc.

**Q7. Does FastAPI work with sync & async**  
**A.** Yes; you can mix `def` and `async def` endpoints.

**Q8. Built‑in dev server?**  
**A.** Use Uvicorn: `uvicorn main:app --reload`.

**Q9. How does typing improve DX?**  
**A.** Editor autocompletion, early errors, accurate docs & validation.

**Q10. How to set app metadata?**  
**A.** Title, version, description in `FastAPI(...)`.

```python
app = FastAPI(title="Shop API", version="1.0.0", description="Demo")
```

---

## 2) Routing & Path Operations (11–20)

**Q11. Define a GET with a path parameter.**  
```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

**Q12. Order of routes with same prefix matters?**  
**A.** Yes—more specific routes should be defined before generic ones.

**Q13. Use `APIRouter` and include it.**  
```python
from fastapi import APIRouter, Depends
router = APIRouter(prefix="/users", tags=["users"])

@router.get("/{user_id}")
def get_user(user_id: int): return {"user_id": user_id}

app.include_router(router)
```

**Q14. Add route tags & summary.**  
```python
from fastapi import FastAPI
@app.get("/health", tags=["infra"], summary="Health check")
def health(): return {"status": "ok"}
```

**Q15. Route versioning pattern.**  
**A.** Use prefixes like `/v1`, `/v2` via routers.

**Q16. Handle trailing slashes?**  
**A.** Be consistent; `/path` and `/path/` are distinct unless configured.

**Q17. Mount sub‑apps (e.g., static).**  
```python
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
```

**Q18. Include multiple routers with dependencies.**  
```python
from fastapi import Depends
app.include_router(router, dependencies=[Depends(lambda: print("global dep"))])
```

**Q19. Path param converters?**  
**A.** Use type hints (`int`, `UUID`) or Pydantic validators.

**Q20. Route operation IDs?**  
**A.** Auto‑generated; can override via `operation_id` in decorator.

---

## 3) Parameters & Requests (21–30)

**Q21. Query vs Path param difference?**  
**A.** Path is part of URL; query is after `?` used for filtering/paging.

```python
@app.get("/search")
def search(q: str | None = None, limit: int = 10): ...
```

**Q22. Optional parameters.**  
```python
from typing import Optional
@app.get("/opt")
def opt(x: Optional[int] = None): return x
```

**Q23. Multiple values for same key.**  
```python
from typing import List
from fastapi import Query

@app.get("/tags")
def tags(t: List[str] = Query(default=[])): return t
```

**Q24. Form data.**  
```python
from fastapi import Form
@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)): ...
```

**Q25. File uploads.**  
```python
from fastapi import File, UploadFile

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    return {"filename": file.filename}
```

**Q26. Access raw Request.**  
```python
from fastapi import Request
@app.post("/raw")
async def raw(req: Request): return {"len": len(await req.body())}
```

**Q27. Header & Cookie params.**  
```python
from fastapi import Header, Cookie
@app.get("/hdr")
def hdr(ua: str = Header(None), session: str = Cookie(None)): ...
```

**Q28. Aliases & deprecated params.**  
```python
from fastapi import Query
@app.get("/alias")
def alias(user_id: int = Query(..., alias="userId", deprecated=True)): ...
```

**Q29. Default values & constraints.**  
```python
from fastapi import Query
@app.get("/page")
def page(limit: int = Query(10, ge=1, le=100)): ...
```

**Q30. Datetime/UUID params.**  
```python
from uuid import UUID
from datetime import datetime
@app.get("/cast")
def cast(id: UUID, ts: datetime): return {"id": str(id), "ts": ts.isoformat()}
```

---

## 4) Validation & Pydantic (31–40)

**Q31. Basic model with constraints.**  
```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    price: float = Field(gt=0)
```

**Q32. Nested models.**  
```python
class Cart(BaseModel):
    items: list[Item]
```

**Q33. Field validators (Pydantic v2 style).**  
```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    email: str
    @field_validator("email")
    @classmethod
    def check_email(cls, v):
        assert "@" in v, "invalid email"
        return v
```

**Q34. Model config (v2).**  
```python
class ConfigExample(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}
```

**Q35. Enums in models.**  
```python
from enum import Enum
from pydantic import BaseModel
class Status(str, Enum): active="active"; blocked="blocked"
class Account(BaseModel): status: Status
```

**Q36. Union/Annotated types.**  
```python
from typing import Annotated
from pydantic import Field
Amount = Annotated[float, Field(ge=0, le=9999)]
```

**Q37. Custom types.**  
```python
from pydantic import BaseModel
class HexColor(str): ...
class Theme(BaseModel): color: HexColor
```

**Q38. Response_model to limit fields.**  
```python
from pydantic import BaseModel
class User(BaseModel): id: int; name: str; email: str
class PublicUser(BaseModel): id: int; name: str
@app.post("/users", response_model=PublicUser)
def create_user(u: User): return u
```

**Q39. `exclude_none` / `exclude_unset`.**  
```python
from pydantic import BaseModel
class User(BaseModel): id: int; name: str | None = None
u = User(id=1); u.model_dump(exclude_none=True)
```

**Q40. Strict types.**  
```python
from pydantic import BaseModel
class Strict(BaseModel):
    model_config = {"strict": True}
    qty: int
```

---

## 5) Responses & Serialization (41–50)

**Q41. Set status codes.**  
```python
from fastapi import status
@app.post("/create", status_code=status.HTTP_201_CREATED)
def create(): return {"ok": True}
```

**Q42. Response classes.**  
```python
from fastapi.responses import HTMLResponse, PlainTextResponse

@app.get("/html", response_class=HTMLResponse)
def html(): return "<b>Hello</b>"
```

**Q43. Custom headers.**  
```python
from fastapi import Response
@app.get("/hdrs")
def hdrs(resp: Response): resp.headers["X-Trace"]="abc"; return {}
```

**Q44. Streaming response.**  
```python
from fastapi.responses import StreamingResponse

def iter_lines():
    for i in range(3): yield f"{i}\n"
@app.get("/stream")
def stream(): return StreamingResponse(iter_lines(), media_type="text/plain")
```

**Q45. File response.**  
```python
from fastapi.responses import FileResponse
@app.get("/download")
def dl(): return FileResponse("report.pdf", filename="report.pdf")
```

**Q46. Redirects.**  
```python
from fastapi.responses import RedirectResponse
@app.get("/go")
def go(): return RedirectResponse(url="/new")
```

**Q47. Response model encoders.**  
```python
from fastapi.encoders import jsonable_encoder
jsonable_encoder({"dt": object()})  # example; register custom encoders if needed
```

**Q48. Caching headers (manual).**  
```python
from fastapi import Response
@app.get("/cache")
def cache(resp: Response):
    resp.headers["Cache-Control"]="public, max-age=60"; return {"cached": True}
```

**Q49. Gzip/Brotli.**  
```python
from starlette.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware)
```

**Q50. ETags (manual).**  
```python
from fastapi import Response
@app.get("/etag")
def etag(resp: Response): resp.headers["ETag"]='"abc"'; return {"v":1}
```

---

## 6) Dependencies & DI (51–60)

**Q51. What is `Depends`?**  
```python
from fastapi import Depends, Header
def get_token(x_token: str = Header(...)): return x_token
@app.get("/secure")
def secure(token: str = Depends(get_token)): return {"token": token}
```

**Q52. Dependency scopes (yield for teardown).**  
```python
def get_db():
    db = object()
    try: yield db
    finally: print("close db")
```

**Q53. Sub‑dependencies.**  
```python
def a(): return "a"
def b(x: str = Depends(a)): return x + "b"
```

**Q54. Global dependencies.**  
```python
app = FastAPI(dependencies=[Depends(lambda: None)])
```

**Q55. Parameterized dependencies.**  
```python
def limiter(limit:int):
    def inner(): return limit
    return inner

@app.get("/limited")
def limited(l: int = Depends(limiter(10))): return {"limit": l}
```

**Q56. Optional dependency.**  
```python
from fastapi import Security
token: str | None = Security(get_token, scopes=[])
```

**Q57. Context managers with yield.**  
**A.** Open/close resources per request gracefully.

**Q58. Raising in dependencies.**  
```python
from fastapi import HTTPException
def must_admin(role: str = "user"):
    if role != "admin": raise HTTPException(403)
```

**Q59. Caching dependency results.**  
**A.** Called once per unique set of param values per request.

**Q60. Using dependency in router include.**  
```python
app.include_router(router, dependencies=[Depends(lambda: print("check"))])
```

---

## 7) Auth & Security (61–72)

**Q61. OAuth2 Password flow.**  
```python
from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/me")
def me(token: str = Depends(oauth2_scheme)): return {"token": token}
```

**Q62. Issue JWT token.**  
```python
import jwt, time
SECRET="secret"
def create_token(sub:str): 
    return jwt.encode({"sub":sub, "exp": int(time.time())+3600}, SECRET, algorithm="HS256")
```

**Q63. Verify JWT & scopes.**  
```python
from fastapi import Security, Depends
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
oauth2 = OAuth2PasswordBearer(tokenUrl="token", scopes={"admin":"Admin access"})
def get_current_user(scopes: SecurityScopes, token: str = Depends(oauth2)): ...
@app.get("/admin")
def admin(user=Security(get_current_user, scopes=["admin"])): ...
```

**Q64. API Key via header/query.**  
```python
from fastapi import Header, HTTPException
API_KEY="k"
def api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY: raise HTTPException(401)
```

**Q65. Cookie auth example.**  
```python
from fastapi import Response
@app.post("/login")
def login(resp: Response):
    resp.set_cookie("session", "token", httponly=True, samesite="lax")
```

**Q66. CSRF basics?**  
**A.** Use anti‑CSRF token + SameSite cookies for browser‑based POST/PUT/DELETE.

**Q67. Password hashing.**  
```python
from passlib.context import CryptContext
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
hash_ = pwd.hash("secret"); pwd.verify("secret", hash_)
```

**Q68. Rate limiting pattern.**  
**A.** Dependency + Redis counter per IP/user with TTL.

**Q69. CORS configuration.**  
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

**Q70. HTTPS enforcement.**  
**A.** Terminate TLS at proxy; use HSTS and secure cookies.

**Q71. Security best practices.**  
**A.** Short‑lived tokens, rotate secrets, validate inputs, limit upload sizes.

**Q72. Auth in WebSockets?**  
**A.** Validate token on connect; close with policy code if invalid.

---

## 8) Middleware & Lifespan (73–78)

**Q73. What is middleware?**  
```python
from starlette.middleware.base import BaseHTTPMiddleware
class TimerMW(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request); return resp
app.add_middleware(TimerMW)
```

**Q74. Built‑in middlewares.**  
**A.** CORS, GZip, TrustedHost, HTTPSRedirect, SessionMiddleware.

**Q75. Request/response logging.**  
**A.** Implement in middleware or dependency.

**Q76. Lifespan events.**  
```python
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
app = FastAPI(lifespan=lifespan)
```

**Q77. Startup tasks (DB connect).**  
**A.** Open DB pool in lifespan; close on shutdown.

**Q78. Global state.**  
**A.** `app.state` usage for caches/clients.

---

## 9) Background Tasks & Scheduling (79–83)

**Q79. BackgroundTask usage.**  
```python
from fastapi import BackgroundTasks
def send_email(to: str): print("email to", to)
@app.post("/notify")
def notify(to: str, bg: BackgroundTasks):
    bg.add_task(send_email, to); return {"queued": True}
```

**Q80. When to use Celery/RQ instead?**  
**A.** For durability, retries, scheduling, heavy/long jobs.

**Q81. Async background work?**  
**A.** You can add async callables; prefer task runners for resilience.

**Q82. Streaming + background.**  
**A.** Stream response while task proceeds; ensure idempotency.

**Q83. Scheduling cron jobs.**  
**A.** APScheduler/Temporal/Celery beat outside request cycle.

---

## 10) Error Handling (84–89)

**Q84. Use `HTTPException`.**  
```python
from fastapi import HTTPException
# if not found: 
#   raise HTTPException(status_code=404, detail="Not found")
```

**Q85. Custom exception handler.**  
```python
from fastapi.responses import JSONResponse
class AppError(Exception): ...
@app.exception_handler(AppError)
def handle(_, exc: AppError):
    return JSONResponse(status_code=400, content={"error": str(exc)})
```

**Q86. Validation error shape.**  
**A.** `{"detail": [{"loc": ..., "msg": ..., "type": ...}]}`.

**Q87. Problem+JSON responses.**  
**A.** Return RFC7807 structure with correct content-type.

**Q88. Logging exceptions.**  
**A.** Middleware/handlers + `logging` config.

**Q89. Global fallback handler.**  
**A.** Avoid leaking stack traces; return generic 500 with ID.

---

## 11) Async & Concurrency (90–95)

**Q90. When to use `async def`?**  
**A.** I/O bound (DB, HTTP, files); not CPU bound.

**Q91. Mixing sync & async.**  
**A.** Sync runs in threadpool; avoid blocking calls.

**Q92. `asyncio.gather` for parallel I/O.**  
```python
import asyncio
@app.get("/parallel")
async def parallel():
    a, b = await asyncio.gather(fetch_a(), fetch_b())
    return {"a": a, "b": b}
```

**Q93. Avoid blocking calls.**  
**A.** Use async drivers or `run_in_executor` for legacy code.

**Q94. Connection pools.**  
**A.** Reuse clients/DB pools opened at startup.

**Q95. Backpressure.**  
**A.** Tune pool sizes & timeouts; reject overload early.

---

## 12) OpenAPI & Docs (96–100)

**Q96. Where are docs served?**  
**A.** `/docs` and `/redoc`.

**Q97. Customize title/desc/version.**  
**A.** In `FastAPI(...)` constructor metadata.

**Q98. Hide an endpoint.**  
```python
@app.get("/internal", include_in_schema=False)
def internal(): ...
```

**Q99. Override OpenAPI schema.**  
```python
from fastapi.openapi.utils import get_openapi
def custom_openapi():
    if app.openapi_schema: return app.openapi_schema
    app.openapi_schema = get_openapi(title="X", version="1.0", routes=app.routes)
    return app.openapi_schema
app.openapi = custom_openapi
```

**Q100. Add examples to params/body.**  
```python
from pydantic import BaseModel
from fastapi import Body
class Msg(BaseModel): text: str
@app.post("/echo")
def echo(m: Msg = Body(..., examples={"sample":{"summary":"hi","value":{"text":"hello"}}})): return m
```

---

## 13) Testing (101–108)

**Q101. Use `TestClient`.**  
```python
from fastapi.testclient import TestClient
client = TestClient(app)
def test_ping(): assert client.get("/ping").json() == {"ok": True}
```

**Q102. Override dependencies in tests.**  
```python
def fake_db(): return "fake"
app.dependency_overrides[get_db] = fake_db
```

**Q103. Auth in tests.**  
```python
def test_auth():
    r = client.get("/me", headers={"Authorization": "Bearer token"})
```

**Q104. Pytest fixtures for app/db.**  
**A.** Use `yield` fixtures to set up/tear down DB and overrides.

**Q105. Testing validation errors.**  
**A.** Assert status codes & `detail` structure.

**Q106. Using httpx AsyncClient.**  
```python
import pytest, httpx
@pytest.mark.asyncio
async def test_async():
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/ping"); assert r.status_code == 200
```

**Q107. Test WebSockets.**  
```python
def test_ws():
    with client.websocket_connect("/ws") as ws:
        ws.send_text("hi"); assert ws.receive_text() == "hi"
```

**Q108. Load/perf smoke tests.**  
**A.** Use k6/Locust; assert SLOs (p95 latency, error rate).

---

## 14) DB, ORMs & Transactions (109–118)

**Q109. SQLAlchemy sync session pattern.**  
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
engine = create_engine("sqlite:///db.sqlite3", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()
```

**Q110. SQLAlchemy async engine.**  
```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
engine = create_async_engine("sqlite+aiosqlite:///db.sqlite3")
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
```

**Q111. Pydantic with ORM (from_attributes).**  
```python
from pydantic import BaseModel
class UserOut(BaseModel):
    id: int; name: str
    model_config = {"from_attributes": True}
```

**Q112. Alembic migrations.**  
**A.** `alembic init`, set `target_metadata`, `alembic revision --autogenerate`, `alembic upgrade head`.

**Q113. Transactions pattern.**  
```python
def repo(db = Depends(get_db)):
    with db.begin(): ...
```

**Q114. N+1 query avoidance.**  
**A.** Use `joinedload`/`selectinload` and prefetch related entities.

**Q115. Repository pattern.**  
**A.** Encapsulate data access behind interfaces for testability.

**Q116. Pagination.**  
```python
def paginate(q, page:int, size:int): return q.offset((page-1)*size).limit(size)
```

**Q117. Connection pooling.**  
**A.** Configure pool size & timeouts in engine/driver.

**Q118. Async drivers.**  
**A.** `asyncpg`, `aiomysql`, `aiosqlite` with SQLAlchemy async.

---

## 15) Performance, Config & Deployment (119–130)

**Q119. Run in dev.**  
```bash
uvicorn main:app --reload
```

**Q120. Run in prod (Gunicorn + Uvicorn workers).**  
```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 main:app
```

**Q121. Worker count guidance.**  
**A.** Start with `CPU cores * 2` for sync; benchmark for async.

**Q122. Keep‑alive & timeouts.**  
**A.** Tune server/proxy (Gunicorn/Uvicorn/NGINX).

**Q123. Env config.**  
```python
from pydantic_settings import BaseSettings
class Settings(BaseSettings): db_url: str
settings = Settings()  # reads env vars
```

**Q124. Logging config.**  
```python
import logging
logging.basicConfig(level=logging.INFO)
```

**Q125. Profiling.**  
**A.** Use PySpy/yappi; timing middleware; flame graphs.

**Q126. Caching with Redis.**  
```python
# create pool on startup; get/set within routes
```

**Q127. Health & readiness probes.**  
**A.** `/healthz` quick; `/readyz` checks dependencies.

**Q128. Containerization.**  
**A.** Multi‑stage Dockerfile, non‑root, pinned deps.

**Q129. Observability.**  
**A.** Prometheus metrics, OpenTelemetry traces, structured logs.

**Q130. Graceful shutdown.**  
**A.** Handle SIGTERM; close pools in lifespan.

---

## 16) WebSockets & Streaming (131–136)

**Q131. Basic WebSocket echo.**  
```python
from fastapi import WebSocket
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    data = await ws.receive_text()
    await ws.send_text(data)
```

**Q132. Broadcast to many.**  
**A.** Track connections in a set; send messages in a loop.

**Q133. Auth in WS.**  
**A.** Validate token on connect; close with code 1008 if unauthorized.

**Q134. Server‑Sent Events (SSE).**  
**A.** Use `StreamingResponse` with `text/event-stream` and `yield "data: ...\n\n"`.

**Q135. Backpressure in streaming.**  
**A.** Chunk responses; respect client disconnects.

**Q136. Binary frames.**  
**A.** Use `receive_bytes()` / `send_bytes()` for WS binary.

---

## 17) Project Structure & Patterns (137–144)

**Q137. Suggested layout.**  
```
app/
  main.py
  api/routers/*.py
  core/config.py
  models/*.py
  schemas/*.py
  services/*.py
  deps/*.py
  tests/
```

**Q138. Settings module.**  
**A.** Centralize env & constants in `core/config.py`.

**Q139. Service layer.**  
**A.** Put business logic in services, keep routes thin.

**Q140. DTOs vs ORM models.**  
**A.** Separate Pydantic schemas from ORM models.

**Q141. Feature flags.**  
**A.** Inject via settings; toggle in dependencies.

**Q142. API versioning.**  
**A.** Routers per version: `api/v1`, `api/v2`.

**Q143. Idempotency keys.**  
**A.** Header + dedupe store (Redis) for POSTs.

**Q144. Error code catalog.**  
**A.** Central enum map; consistent messages & types.

---

## 18) Integrations & Misc (145–152)

**Q145. Call external APIs (httpx).**  
```python
import httpx
@app.get("/ext")
async def ext():
    async with httpx.AsyncClient() as c:
        r = await c.get("https://example.com")
        return {"status": r.status_code}
```

**Q146. Serving static & SPA.**  
**A.** Mount `StaticFiles`; SPA fallback route.

**Q147. File size limits.**  
**A.** Enforce via proxy (NGINX) and code checks.

**Q148. Multipart + JSON together.**  
**A.** Use `Form`/`File` or send JSON in a form field.

**Q149. Internationalization (i18n).**  
**A.** Add locale header + translation layer.

**Q150. S3/GCS uploads.**  
**A.** Pre‑signed URLs; direct‑to‑bucket uploads.

**Q151. GraphQL with FastAPI.**  
**A.** Mount Strawberry/Graphene ASGI app.

**Q152. Task queues (Celery) integration.**  
**A.** Produce jobs in routes; workers consume, store results.
