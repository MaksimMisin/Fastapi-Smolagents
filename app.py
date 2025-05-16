import os
import asyncio
import time
import threading
import contextvars
import dataclasses
import json
from typing import Optional, List, Dict, Any, Callable, Coroutine, AsyncGenerator
import functools
from smolagents import CodeAgent, WebSearchTool, InferenceClientModel, Tool
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
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
    send_stream_response: Optional[Callable[[Dict[str, Any]], None]] = None
    data_queue: Optional[asyncio.Queue[Optional[Dict[str, Any]]]] = None


request_context_var: contextvars.ContextVar[RequestContext] = contextvars.ContextVar(
    "request_context_var"
)


class RequestContextAccessor:
    """Provides safe access to the request context and its attributes."""

    _CONTEXT_VAR = request_context_var

    @staticmethod
    def get_context() -> RequestContext:
        """Retrieves the current RequestContext."""
        return RequestContextAccessor._CONTEXT_VAR.get()


def run_in_main_loop(
    coroutine_callable: Callable[[], Coroutine[Any, Any, Any]],
    timeout: float = 15.0,
):
    """
    Runs an async function (via a callable that produces its coroutine, e.g., a partial)
    synchronously in the main event loop.
    """
    event_loop = RequestContextAccessor.get_context().event_loop

    try:
        coroutine = coroutine_callable()
    except Exception as e:
        print(
            f"[{threading.current_thread().name}] run_in_main_loop: Error creating coroutine: {e}"
        )
        return {"error": f"Error creating coroutine: {str(e)}"}

    future = asyncio.run_coroutine_threadsafe(coroutine, event_loop)

    try:
        return future.result(timeout=timeout)
    except asyncio.TimeoutError:
        request_id_str = "unknown"
        request_id_str = RequestContextAccessor.get_context().request_id
        print(
            f"[{threading.current_thread().name}] run_in_main_loop: Timeout waiting for coroutine_callable (context: {request_id_str})"
        )
        return {
            "error": f"Timeout waiting for coroutine_callable (context: {request_id_str})",
            "timeout": True,
        }
    except Exception as e:
        print(f"[{threading.current_thread().name}] run_in_main_loop: Error")
        return {
            "error": f"Error running coroutine_callable: {str(e)}",
        }


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

    async def db_call(self, item_id: int):
        """Asynchronously fetches an item from the database."""
        current_session_maker = RequestContextAccessor.get_context().db_session_maker

        try:
            async with current_session_maker() as session:
                result = await session.execute(select(Item).where(Item.id == item_id))
                item = result.scalars().first()
                if item:
                    return item.to_dict()
                else:
                    return {
                        "error": f"Item with ID {item_id} not found.",
                        "id": item_id,
                        "found": False,
                    }
        except Exception as e:
            return {
                "error": f"Database error while fetching item {item_id}: {str(e)}",
                "id": item_id,
            }

    def forward(self, item_id: int):
        """Synchronously fetches an item by creating a partial of the async db_call method
        and running it in the main event loop. Streams the result if in a streaming context.
        """
        db_call_partial = functools.partial(self.db_call, item_id=item_id)
        result = run_in_main_loop(
            coroutine_callable=db_call_partial,
            timeout=15.0,
        )

        try:
            request_ctx = RequestContextAccessor.get_context()
            send_stream_response_func = request_ctx.send_stream_response
            if send_stream_response_func:
                event_data = {
                    "type": "tool_result",
                    "tool_name": self.name,
                    "item_id": item_id,
                    "data": result,
                }
                send_stream_response_func(event_data)
        except Exception as e:
            print(
                f"[{threading.current_thread().name}] Error streaming tool result: {e}"
            )

        return result


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

    # If this is a request to the streaming endpoint, set up the stream
    if request.url.path.startswith("/process_item_smolagents/"):
        queue, sender_callable = setup_request_stream()
        request_ctx.data_queue = queue
        request_ctx.send_stream_response = sender_callable

    token = request_context_var.set(request_ctx)

    try:
        response = await call_next(request)
    finally:
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
    request_ctx = RequestContextAccessor.get_context()
    session_maker = request_ctx.db_session_maker
    request_id_str = request_ctx.request_id
    print(
        f"[{thread_name}] --> Executing async DB query for ID: {item_id} (Request ID: {request_id_str})"
    )

    try:
        async with session_maker() as session:
            result = await session.execute(select(Item).where(Item.id == item_id))
            item = result.scalars().first()
            print(f"[{thread_name}] <-- Finished async DB query for ID: {item_id}")
            return item
    except Exception as e:
        print(f"[{thread_name}] Error during async DB query: {e}")
        raise


def cpu_heavy_operation_with_db(item_id: int):
    """
    A synchronous function simulating a CPU-heavy task that also
    needs to fetch data from the database.
    This function is intended to run in a separate thread from the main event loop.
    It reads the RequestContext and send_stream_response callable from context variables.
    """
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Starting streaming CPU-heavy operation for ID: {item_id}")
    start_time = time.time()

    request_ctx = RequestContextAccessor.get_context()
    send_stream_response = request_ctx.send_stream_response

    try:
        db_fetcher_tool = DatabaseItemFetcherTool()
        model = InferenceClientModel()
        agent = CodeAgent(
            tools=[WebSearchTool(), db_fetcher_tool], model=model, stream_outputs=True
        )

        agent_prompt = (
            f"First, use the 'database_item_fetcher' tool to get details for item ID {item_id}. "
            "Then, based on the item's description (if found), tell me a short story about it. "
            "If the item is not found or there's an error, state that you could not retrieve the item."
        )

        print(f"[{thread_name}] Starting agent.run for ID: {item_id}")
        for chunk in agent.run(agent_prompt):
            send_stream_response({"type": "update", "content": chunk})

        cpu_time = time.time() - start_time
        print(
            f"[{thread_name}] Finished agent work in {cpu_time:.4f} seconds for ID: {item_id}"
        )

    except Exception as e:
        print(f"[{thread_name}] Error during CPU-heavy operation for ID {item_id}: {e}")
        try:
            send_stream_response(
                {"type": "error", "message": str(e), "item_id": item_id}
            )  # type: ignore
        except Exception as send_e:
            print(
                f"[{thread_name}] Critical error: Failed to send error to stream: {send_e}"
            )
        raise
    finally:
        print(
            f"[{thread_name}] Synchronous CPU-heavy operation finished for ID: {item_id}."
        )


# --- Stream Utilities ---


def setup_request_stream() -> (
    tuple[asyncio.Queue[Optional[Dict[str, Any]]], Callable[[Dict[str, Any]], None]]
):
    """
    Creates an asyncio.Queue and a thread-safe callable to put items onto it.
    The callable uses run_in_main_loop to ensure queue.put is run in the main event loop.
    """
    queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    async def _async_put_item_on_queue(item: Dict[str, Any]):
        await queue.put(item)

    def send_response_callable(event_data: Dict[str, Any]):
        run_in_main_loop(
            functools.partial(_async_put_item_on_queue, event_data), timeout=5.0
        )

    return queue, send_response_callable


async def generate_ndjson_stream(
    queue: asyncio.Queue[Optional[Dict[str, Any]]],
) -> AsyncGenerator[str, None]:
    """
    Async generator that consumes items from a queue and yields them as NDJSON strings.
    Terminates when None is received from the queue.
    """
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        try:
            yield json.dumps(item) + "\n"
        except TypeError as e:
            error_event = {
                "type": "error",
                "message": f"Stream serialization error: {str(e)}",
                "problematic_item_type": str(type(item)),
            }
            yield json.dumps(error_event) + "\n"
        finally:
            queue.task_done()


# --- Stream Processing Helper Functions ---


async def _put_final_status_on_queue(
    item_id: int,
    success: bool,
    detail: Optional[str] = None,
):
    """Puts a 'done' status message onto the data_queue fetched from context."""
    # Let LookupError propagate if context_var is not set.
    data_queue = RequestContextAccessor.get_context().data_queue
    # If data_queue is None (e.g. context was found but queue not set up for this request type),
    # the await data_queue.put() will raise an AttributeError. This is "fail naturally".
    # The check for `if not data_queue:` was already removed in a previous step.

    if success:
        await data_queue.put({"type": "done", "status": "success", "item_id": item_id})
    else:
        await data_queue.put(
            {
                "type": "done",
                "status": "error",
                "detail": detail or "Unknown error",
                "item_id": item_id,
            }
        )


async def _run_cpu_op_and_manage_queue(item_id: int):
    """
    Runs the CPU-heavy operation in a thread, puts results/status on the data_queue (fetched from context),
    and finally puts None on the queue to signal completion to consumers.
    (This function is now at the module level)
    """
    current_context = contextvars.copy_context()
    try:
        # cpu_heavy_operation_with_db is expected to use send_stream_response
        # (which uses run_in_main_loop to put on data_queue) for its intermediate updates.
        await asyncio.to_thread(
            current_context.run, cpu_heavy_operation_with_db, item_id
        )
        # If cpu_heavy_operation_with_db completes without raising an exception that it
        # didn't already handle by sending an error event, queue success status.
        await _put_final_status_on_queue(item_id, success=True)
    except Exception as e:
        # This catches exceptions from cpu_heavy_operation_with_db if it re-raises,
        # or errors during its execution in the thread if not caught internally.
        print(
            f"[{threading.current_thread().name}] Background operation for item {item_id} encountered an error: {e}"
        )
        try:
            await _put_final_status_on_queue(item_id, success=False, detail=str(e))
        except Exception as q_e:
            # If queueing the error status itself fails (e.g. queue not found in context by accessor),
            # this is a more critical problem.
            print(
                f"[{threading.current_thread().name}] CRITICAL: Failed to queue error status for item {item_id} after operation error: {q_e}"
            )
            # This exception will propagate out of this function
            # and be caught when the task running this coroutine is awaited.
            raise
    finally:
        # Always ensure None is put on the queue to signal the end of data
        # for generate_ndjson_stream, regardless of success or failure above.
        try:
            request_ctx = RequestContextAccessor.get_context()
            data_queue = request_ctx.data_queue
            if data_queue:
                await data_queue.put(None)
            else:
                print(
                    f"[{threading.current_thread().name}] CRITICAL: Data queue not found in context for putting None sentinel (item {item_id})"
                )
        except LookupError as e:
            print(
                f"[{threading.current_thread().name}] CRITICAL: Failed to get context for None sentinel (item {item_id}): {e}"
            )
        except Exception as e:
            print(
                f"[{threading.current_thread().name}] CRITICAL: Failed to put None sentinel on data_queue for item {item_id}: {e}"
            )
            # This is a critical failure in the streaming mechanism.
            # This exception will also propagate if not already handling one.


async def _event_stream_generator(item_id: int) -> AsyncGenerator[str, None]:
    """
    The async generator returned to the client. It starts the background operation
    and then yields items from the data_queue (fetched from context).
    (This function is now at the module level)
    """
    operation_task: Optional[asyncio.Task] = None
    try:
        request_ctx = RequestContextAccessor.get_context()
        data_queue = request_ctx.data_queue

        operation_task = asyncio.create_task(_run_cpu_op_and_manage_queue(item_id))

        # Consume items from the data_queue via generate_ndjson_stream.
        # This will yield all intermediate messages and the final status message.
        # It terminates when _run_cpu_op_and_manage_queue puts None on the data_queue.
        async for item_json_str in generate_ndjson_stream(data_queue):
            yield item_json_str

        # After the stream is exhausted, await the operation task to ensure it finished
        # and to propagate any exceptions that occurred within _run_cpu_op_and_manage_queue
        # itself (e.g., if putting to queue failed critically).
        if operation_task:  # Should always be true if create_task succeeded
            await operation_task

    except asyncio.CancelledError:
        print(
            f"[{threading.current_thread().name}] Event stream generator for item {item_id} was cancelled."
        )
        if operation_task and not operation_task.done():
            operation_task.cancel()
            try:
                await operation_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(
                    f"[{threading.current_thread().name}] Error during cancellation of operation task for {item_id}: {e}"
                )
        raise
    except Exception as e:
        print(
            f"[{threading.current_thread().name}] Error in event_stream_generator for item {item_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Streaming failed for item {item_id}: {str(e)}"
        )
    finally:
        print(
            f"[{threading.current_thread().name}] Event stream generator for item {item_id} finishing."
        )
        # Ensure the background task is awaited if it exists and hasn't been,
        # especially if an unexpected error caused an early exit from the try block.
        if operation_task and not operation_task.done():
            print(
                f"[{threading.current_thread().name}] Cleaning up operation task for {item_id} in finally block."
            )
            try:
                await operation_task
            except Exception as e:
                # Log if it failed during this final await, but don't overshadow original exception if one exists.
                print(
                    f"[{threading.current_thread().name}] Error while awaiting operation task in finally for {item_id}: {e}"
                )


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
        request_id_str = RequestContextAccessor.get_context().request_id
        print(
            f"[{threading.current_thread().name}] ContextVar Request ID: {request_id_str}"
        )
    except LookupError:  # Catch if RequestContextAccessor.get_context() fails
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

    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Received request to create item: {name}")

    try:
        request_ctx = RequestContextAccessor.get_context()
        session_maker = request_ctx.db_session_maker
        request_id_str = request_ctx.request_id
        print(
            f"[{thread_name}] Using session_maker from context (Request ID: {request_id_str})"
        )
    except LookupError as e:
        print(f"[{thread_name}] Error: {e} for create_item_endpoint!")
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

    try:
        request_ctx = RequestContextAccessor.get_context()
        session_maker = request_ctx.db_session_maker
        request_id_str = request_ctx.request_id
        print(
            f"[{thread_name}] Using session_maker from context (Request ID: {request_id_str})"
        )
    except LookupError as e:
        print(f"[{thread_name}] Error: {e} for read_items_endpoint!")
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


@app.get("/process_item_smolagents/{item_id}")
async def process_item_smolagents_endpoint(item_id: int):
    """
    Endpoint to trigger a CPU-heavy operation in a separate thread, streaming results.
    Uses ContextVars to pass a stream-sending callable to the threaded operation.
    """
    thread_name = threading.current_thread().name
    print(
        f"[{thread_name}] Received request to stream process item CPU heavy for ID: {item_id}"
    )

    streaming_generator = _event_stream_generator(item_id)

    return StreamingResponse(streaming_generator, media_type="application/x-ndjson")
