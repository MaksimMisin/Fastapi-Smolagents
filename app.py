import os
import asyncio
import time
import threading
import contextvars
import dataclasses
from typing import Optional, List, Dict, Any
import functools
from smolagents import CodeAgent, WebSearchTool, InferenceClientModel, Tool
from fastapi import FastAPI, HTTPException, Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy.future import select
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base


POOL_SIZE = 10
DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/testdb"

Base = declarative_base()

app = FastAPI()


@dataclasses.dataclass
class RequestContext:
    """Holds context data for a single request."""

    db_engine: AsyncEngine
    db_session_maker: async_sessionmaker
    request_id: str
    event_loop: asyncio.AbstractEventLoop


request_context_var: contextvars.ContextVar[RequestContext] = contextvars.ContextVar(
    "request_context_var"
)


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column("name", String, index=True)
    description = Column("description", String, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Helper to convert Item object to dictionary."""
        return {"id": self.id, "name": self.name, "description": self.description}


class DatabaseItemFetcherTool(Tool):
    name = "database_item_fetcher"
    description = "Fetches an item's details (ID, name, description) from the application's database using its ID"
    inputs = {
        "item_id": {
            "type": "integer",
            "description": "The unique identifier of the item to fetch.",
        }
    }
    output_type = "object"

    def __init__(self):
        super().__init__()

    def forward(self, item_id: int):
        async def _async_fetch_item_helper(id_to_fetch: int):
            current_thread_name = threading.current_thread().name
            print(
                f"[{current_thread_name}] Tool's _async_fetch_item_helper: Fetching item {id_to_fetch}"
            )
            try:
                # This relies on the context being correctly propagated to this thread
                request_ctx = request_context_var.get()
                current_session_maker = request_ctx.db_session_maker
            except LookupError:
                print(
                    f"[{current_thread_name}] Tool's _async_fetch_item_helper: Error - request_context_var not set."
                )
                return {
                    "error": "Context not available for DB operation in tool.",
                    "id": id_to_fetch,
                }

            print(
                f"[{current_thread_name}] Tool's _async_fetch_item_helper: Using session_maker for item {id_to_fetch}"
            )
            try:
                async with current_session_maker() as session:
                    result = await session.execute(
                        select(Item).where(Item.id == id_to_fetch)
                    )
                    item = result.scalars().first()
                    if item:
                        print(
                            f"[{current_thread_name}] Tool's _async_fetch_item_helper: Item {id_to_fetch} found."
                        )
                        return item.to_dict()
                    else:
                        print(
                            f"[{current_thread_name}] Tool's _async_fetch_item_helper: Item {id_to_fetch} not found."
                        )
                        return {
                            "error": f"Item with ID {id_to_fetch} not found.",
                            "id": id_to_fetch,
                            "found": False,
                        }
            except Exception as e:
                print(
                    f"[{current_thread_name}] Tool's _async_fetch_item_helper: DB error for item {id_to_fetch}: {e}"
                )
                return {
                    "error": f"Database error while fetching item {id_to_fetch}: {str(e)}",
                    "id": id_to_fetch,
                }

        try:
            request_ctx = request_context_var.get()
            event_loop = request_ctx.event_loop
        except LookupError:
            print(
                f"[{threading.current_thread().name}] Tool's forward: Error - request_context_var not set when trying to get event_loop."
            )
            return {
                "error": "Context not available for event loop in tool.",
                "id": item_id,
            }

        future = asyncio.run_coroutine_threadsafe(
            _async_fetch_item_helper(item_id), event_loop
        )

        try:
            return future.result(timeout=15.0)
        except asyncio.TimeoutError:
            print(
                f"[{threading.current_thread().name}] Tool's forward: Timeout waiting for DB operation for item ID {item_id}."
            )
            return {
                "error": f"Timeout waiting for database operation for item ID {item_id}.",
                "id": item_id,
                "timeout": True,
            }
        except Exception as e:
            print(
                f"[{threading.current_thread().name}] Tool's forward: Error executing DB fetch for item ID {item_id}: {e}"
            )
            return {
                "error": f"Error executing database fetch for item ID {item_id}: {str(e)}",
                "id": item_id,
            }


def create_engine(
    echo=False, pool_size=POOL_SIZE, application_name=None
) -> Optional[AsyncEngine]:
    """Creates an async SQLAlchemy engine with a connection pool."""
    if application_name is None:
        application_name = os.getenv("APPLICATION_NAME", "demo")

    return create_async_engine(
        DATABASE_URL,
        echo=echo,
        poolclass=AsyncAdaptedQueuePool,
        pool_timeout=30,  # Connection timeout
        max_overflow=10,  # Max connections beyond pool_size
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_size=pool_size,  # Minimum pool size
        connect_args={"server_settings": {"application_name": application_name}},
    )


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    Middleware to create and set the RequestContext in a context variable
    for each request.
    """
    # Retrieve the engine, session maker, and main event loop from app state
    engine = app.state.db_engine
    session_maker = app.state.db_session_maker
    main_event_loop = app.state.main_event_loop

    # Create the RequestContext instance for this request
    # In a real app, you might generate a unique request_id here
    request_ctx = RequestContext(
        db_engine=engine,
        db_session_maker=session_maker,
        request_id=str(time.time()),
        event_loop=main_event_loop,
    )

    # Set the context variable for the current request
    token = request_context_var.set(request_ctx)

    try:
        response = await call_next(request)
    finally:
        # Reset the context variable when the request is finished
        request_context_var.reset(token)

    return response


async def get_item_async(item_id: int) -> Optional[Item]:
    """
    Asynchronous function to get an item by ID from the database.
    This function is designed to be awaited and runs within the event loop
    that owns the session_maker (the main application event loop).
    It retrieves the session maker from the context variable.
    """
    thread_name = threading.current_thread().name
    try:
        # Retrieve the RequestContext from the context variable
        request_ctx = request_context_var.get()
        print(
            f"[{thread_name}] --> Executing async DB query for ID: {item_id} (Request ID: {request_ctx.request_id})"
        )
        session_maker = request_ctx.db_session_maker
    except LookupError:
        print(f"[{thread_name}] Error: request_context_var not set in async context!")
        raise RuntimeError("Request context not available in async context")

    try:
        async with session_maker() as session:
            result = await session.execute(select(Item).where(Item.id == item_id))
            item = result.scalars().first()
            print(f"[{thread_name}] <-- Finished async DB query for ID: {item_id}")
            return item
    except Exception as e:
        print(f"[{thread_name}] Error during async DB query: {e}")
        raise  # Re-raise the exception


# --- Synchronous CPU-heavy operation that needs DB access ---
def cpu_heavy_operation_with_db(item_id: int):  # event_loop parameter removed
    """
    A synchronous function simulating a CPU-heavy task that also
    needs to fetch data from the database.
    This function is intended to run in a separate thread from the main event loop.
    It reads the RequestContext from a context variable.
    """
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Starting synchronous CPU-heavy operation for ID: {item_id}")

    # blocking operation
    start_time = time.time()

    # DatabaseItemFetcherTool no longer needs event_loop in constructor
    db_fetcher_tool = DatabaseItemFetcherTool()

    model = InferenceClientModel()
    # Add the new tool to the agent
    agent = CodeAgent(
        tools=[WebSearchTool(), db_fetcher_tool], model=model, stream_outputs=True
    )

    # Modify the prompt to use the tool
    agent_prompt = (
        f"First, use the 'database_item_fetcher' tool to get details for item ID {item_id}. "
        "Then, based on the item's description (if found), tell me a short story about it. "
        "If the item is not found or there's an error, state that you could not retrieve the item."
    )
    agent_result = agent.run(agent_prompt)

    cpu_time = time.time() - start_time
    print(
        f"[{thread_name}] Finished agent work in {cpu_time:.4f} seconds, agent_result: {agent_result}"
    )

    print(f"[{thread_name}] Synchronous CPU-heavy operation finished.")
    # The function now returns the agent's result directly.
    # The structure of this return value has changed from {"db_item": ...} to {"agent_output": ...}
    return {"cpu_work_duration": cpu_time, "agent_output": agent_result}


# --- Application Lifespan Events ---


async def create_sample_data(session_maker: async_sessionmaker[AsyncSession]):
    """Creates sample data in the database."""
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Attempting to create sample data...")
    async with session_maker() as session:
        async with session.begin():
            # Check if data already exists to avoid duplicates on reload
            result = await session.execute(select(Item).limit(1))
            if result.scalars().first() is not None:
                print(f"[{thread_name}] Sample data already exists. Skipping creation.")
                return

            sample_items = [
                Item(
                    name="Sample Item 1", description="This is the first sample item."
                ),
                Item(
                    name="Sample Item 2", description="Another sample item for testing."
                ),
                Item(
                    name="Sample Item 3",
                    description="A third item to populate the database.",
                ),
            ]
            session.add_all(sample_items)
            await session.commit()
            print(
                f"[{thread_name}] Successfully created {len(sample_items)} sample items."
            )


@app.on_event("startup")
async def startup_db():
    """Initializes the database engine, session maker, and stores the main event loop."""
    print(
        f"[{threading.current_thread().name}] Application startup: Initializing DB engine and session maker."
    )

    engine = create_engine()
    if engine is None:
        raise ConnectionError("Failed to create database engine")

    # Create the session maker factory
    session_maker = async_sessionmaker(engine, class_=AsyncSession, autocommit=False)

    # Store engine and sessionmaker on app state (used by middleware)
    app.state.db_engine = engine
    app.state.db_session_maker = session_maker

    # Optional: Create tables on startup (for demonstration)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print(f"[{threading.current_thread().name}] Database tables checked/created.")
        # Create sample data
        await create_sample_data(session_maker)
    except Exception as e:
        print(
            f"[{threading.current_thread().name}] Error creating tables or sample data: {e}"
        )
        # Depending on your setup, this might be a critical error

    # Store the main event loop reference (needed for run_coroutine_threadsafe)
    # This is the event loop that Uvicorn is running the FastAPI app on.
    app.state.main_event_loop = asyncio.get_event_loop()
    print(
        f"[{threading.current_thread().name}] Main event loop stored: {app.state.main_event_loop}"
    )


@app.on_event("shutdown")
async def shutdown_db():
    """Disposes the database engine and shuts down the thread pool executor."""
    print(
        f"[{threading.current_thread().name}] Application shutdown: Disposing DB engine and closing executor."
    )
    # Dispose of the engine and close connections
    if hasattr(app.state, "db_engine") and app.state.db_engine:
        await app.state.db_engine.dispose()
        print(f"[{threading.current_thread().name}] Database engine disposed.")


# --- FastAPI Endpoints ---


@app.get("/")
async def read_root():
    """Basic root endpoint."""
    print(f"[{threading.current_thread().name}] Received request for root.")
    # Access context var (optional, just for demonstration)
    try:
        req_ctx = request_context_var.get()
        print(
            f"[{threading.current_thread().name}] ContextVar Request ID: {req_ctx.request_id}"
        )
    except LookupError:
        print(
            f"[{threading.current_thread().name}] ContextVar not set for root endpoint."
        )

    return {
        "message": "FastAPI app with asyncpg, SQLAlchemy, threading, and ContextVars example"
    }


@app.post("/items/")
async def create_item_endpoint(name: str, description: Optional[str] = None):
    """Endpoint to create a new item (runs in main async loop)."""
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Received request to create item: {name}")

    # Retrieve the session maker from the context variable
    try:
        request_ctx = request_context_var.get()
        print(
            f"[{thread_name}] Using session_maker from context (Request ID: {request_ctx.request_id})"
        )
        session_maker = request_ctx.db_session_maker
    except LookupError:
        print(
            f"[{thread_name}] Error: request_context_var not set for create_item_endpoint!"
        )
        raise HTTPException(
            status_code=500, detail="Database session maker not available"
        )

    try:
        async with session_maker() as session:
            db_item = Item(name=name, description=description)
            session.add(db_item)
            await session.commit()
            await session.refresh(db_item)
            print(f"[{thread_name}] Item created with ID: {db_item.id}")
            return db_item.to_dict()
    except Exception as e:
        print(f"[{thread_name}] Error creating item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create item: {e}")


@app.get("/items/", response_model=List[Dict[str, Any]])
async def read_items_endpoint(skip: int = 0, limit: int = 10):
    """Endpoint to read items (runs in main async loop)."""
    thread_name = threading.current_thread().name
    print(
        f"[{thread_name}] Received request to read items (skip: {skip}, limit: {limit})."
    )

    # Retrieve the session maker from the context variable
    try:
        request_ctx = request_context_var.get()
        print(
            f"[{thread_name}] Using session_maker from context (Request ID: {request_ctx.request_id})"
        )
        session_maker = request_ctx.db_session_maker
    except LookupError:
        print(
            f"[{thread_name}] Error: request_context_var not set for read_items_endpoint!"
        )
        raise HTTPException(
            status_code=500, detail="Database session maker not available"
        )

    try:
        async with session_maker() as session:
            result = await session.execute(select(Item).offset(skip).limit(limit))
            items = result.scalars().all()
            print(f"[{thread_name}] Retrieved {len(items)} items.")
            return [item.to_dict() for item in items]
    except Exception as e:
        print(f"[{thread_name}] Error reading items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read items: {e}")


@app.get("/process_item_smolagents/{item_id}", response_model=Dict[str, Any])
async def process_item_smolagents_endpoint(item_id: int):
    """
    Endpoint to trigger a CPU-heavy operation in a separate thread
    that also needs to access the database using run_coroutine_threadsafe and ContextVars.
    """
    thread_name = threading.current_thread().name
    print(
        f"[{thread_name}] Received request to process item CPU heavy for ID: {item_id}"
    )

    current_context = contextvars.copy_context()
    try:
        # Wrap the synchronous function call with current_context.run()
        # This ensures that the context (including request_context_var)
        # is copied and available in the target thread.
        result = await asyncio.to_thread(
            functools.partial(current_context.run, cpu_heavy_operation_with_db),
            item_id,  # main_loop is no longer passed here
        )

        print(
            f"[{thread_name}] CPU heavy operation finished in thread, result received."
        )
        return {"status": "success", "data": result}

    except RuntimeError as e:
        print(f"[{thread_name}] Caught runtime error from thread: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    except Exception as e:
        print(f"[{thread_name}] An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
