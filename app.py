import os
import asyncio
import time
import threading
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException
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

# Assume a simple SQLAlchemy model
Base = declarative_base()


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column("name", String, index=True)
    description = Column("description", String, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Helper to convert Item object to dictionary."""
        return {"id": self.id, "name": self.name, "description": self.description}


POOL_SIZE = 10
DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/testdb"


# Database utility functions (provided by the user)
def create_engine(
    echo=False, pool_size=POOL_SIZE, application_name=None
) -> Optional[AsyncEngine]:
    """Creates an async SQLAlchemy engine with a connection pool."""
    if application_name is None:
        application_name = os.getenv("APPLICATION_NAME", "supersimple-python")
    # Fallback database URL for demonstration if env var is not set

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


# FastAPI application instance
app = FastAPI()


# --- Async Coroutine for Database Interaction (runs in the main event loop) ---
async def get_item_async(
    item_id: int, session_maker: async_sessionmaker
) -> Optional[Item]:
    """
    Asynchronous function to get an item by ID from the database.
    This function is designed to be awaited and runs within the event loop
    that owns the session_maker (the main application event loop).
    """
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] --> Executing async DB query for ID: {item_id}")
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
def cpu_heavy_operation_with_db(
    item_id: int,
    event_loop: asyncio.AbstractEventLoop,
    session_maker: async_sessionmaker,
):
    """
    A synchronous function simulating a CPU-heavy task that also
    needs to fetch data from the database.
    This function is intended to run in a separate thread from the main event loop.
    """
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Starting synchronous CPU-heavy operation for ID: {item_id}")

    # Simulate CPU work (blocking operation)
    start_time = time.time()
    result = 0
    for i in range(20_000_000):  # Increased loop for a more noticeable delay
        result += i % 100
    cpu_time = time.time() - start_time
    print(
        f"[{thread_name}] Finished CPU work in {cpu_time:.4f} seconds, partial result: {result}"
    )

    # Now, interact with the database from this synchronous thread.
    # The database interaction is asynchronous (using asyncpg/SQLAlchemy async).
    # We MUST use asyncio.run_coroutine_threadsafe to schedule and run
    # the async DB call (get_item_async) in the main event loop thread
    # where the asyncpg pool is managed.
    print(f"[{thread_name}] Attempting to access DB from synchronous thread...")
    db_item_data = None
    try:
        # Submit the async DB coroutine to the main event loop
        # run_coroutine_threadsafe returns a Future
        future = asyncio.run_coroutine_threadsafe(
            get_item_async(item_id, session_maker),  # The async coroutine to run
            event_loop,  # The target event loop (the main one)
        )
        print(
            f"[{thread_name}] Submitted async DB task to main loop. Waiting for result..."
        )

        # Wait for the async DB operation to complete and get the result
        # This call blocks the current synchronous thread until the future is done.
        db_item = future.result(
            timeout=10.0
        )  # Add a timeout to prevent infinite blocking

        if db_item:
            db_item_data = db_item.to_dict()
        print(f"[{thread_name}] Received result from async DB task: {db_item_data}")

    except asyncio.TimeoutError:
        print(f"[{thread_name}] Timeout waiting for async DB operation.")
        # Handle timeout specifically if needed
        raise RuntimeError("Timeout waiting for database operation")
    except Exception as e:
        print(f"[{thread_name}] Error during DB interaction from sync thread: {e}")
        # Handle exceptions appropriately - re-raise or return error info
        raise RuntimeError(f"DB interaction failed in thread: {e}") from e

    print(f"[{thread_name}] Synchronous CPU-heavy operation finished.")
    return {"cpu_work_duration": cpu_time, "db_item": db_item_data}


# --- Application Lifespan Events ---


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

    # Store engine and sessionmaker on app state for easy access
    app.state.db_engine = engine
    app.state.db_session_maker = session_maker

    # Optional: Create tables on startup (for demonstration)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print(f"[{threading.current_thread().name}] Database tables checked/created.")
    except Exception as e:
        print(f"[{threading.current_thread().name}] Error creating tables: {e}")
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
    return {"message": "FastAPI app with asyncpg, SQLAlchemy, and threading example"}


@app.post("/items/")
async def create_item_endpoint(name: str, description: Optional[str] = None):
    """Endpoint to create a new item (runs in main async loop)."""
    print(
        f"[{threading.current_thread().name}] Received request to create item: {name}"
    )
    session_maker = app.state.db_session_maker
    try:
        async with session_maker() as session:
            db_item = Item(name=name, description=description)
            session.add(db_item)
            await session.commit()
            await session.refresh(db_item)
            print(
                f"[{threading.current_thread().name}] Item created with ID: {db_item.id}"
            )
            return db_item.to_dict()
    except Exception as e:
        print(f"[{threading.current_thread().name}] Error creating item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create item: {e}")


@app.get("/items/", response_model=List[Dict[str, Any]])
async def read_items_endpoint(skip: int = 0, limit: int = 10):
    """Endpoint to read items (runs in main async loop)."""
    print(
        f"[{threading.current_thread().name}] Received request to read items (skip: {skip}, limit: {limit})."
    )
    session_maker = app.state.db_session_maker
    try:
        async with session_maker() as session:
            result = await session.execute(select(Item).offset(skip).limit(limit))
            items = result.scalars().all()
            print(f"[{threading.current_thread().name}] Retrieved {len(items)} items.")
            return [item.to_dict() for item in items]
    except Exception as e:
        print(f"[{threading.current_thread().name}] Error reading items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read items: {e}")


@app.get("/process_item_cpu_heavy/{item_id}", response_model=Dict[str, Any])
async def process_item_cpu_heavy_endpoint(item_id: int):
    """
    Endpoint to trigger a CPU-heavy operation in a separate thread
    that also needs to access the database using run_coroutine_threadsafe.
    """
    thread_name = threading.current_thread().name
    print(
        f"[{thread_name}] Received request to process item CPU heavy for ID: {item_id}"
    )

    # Get the main event loop and session maker from app state
    main_loop = app.state.main_event_loop
    session_maker = app.state.db_session_maker  # Pass the maker to the sync function

    # Run the synchronous CPU-heavy function in the thread pool executor.
    # This offloads the blocking CPU work to a separate thread.
    try:
        # Use asyncio.to_thread (Python 3.9+) or loop.run_in_executor
        # asyncio.to_thread is generally preferred for simplicity
        result = await asyncio.to_thread(
            cpu_heavy_operation_with_db,  # The synchronous function to run in a thread
            item_id,
            main_loop,  # Pass the main event loop reference
            session_maker,  # Pass the session maker factory
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
