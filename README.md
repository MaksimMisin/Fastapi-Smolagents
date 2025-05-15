# FastAPI async db and smolagents example

- **FastAPI Application (`app.py`):**
    - Basic CRUD operations for "Items" stored in a PostgreSQL database.
    - `/process_item_smolagents/{item_id}` endpoint that:
        - Uses a `smolagents.CodeAgent`.
        - Utilizes a `DatabaseItemFetcherTool` to fetch item details from the database. Async database queries are executed in a separate thread, by using `asyncio.run_coroutine_threadsafe` and a `RequestContext` propagated via `contextvars`.
    - Database initialization and sample data creation on startup.

- **Benchmark Script (`benchmark.py`):**
    - Sends multiple requests to the CPU-heavy endpoint (`/process_item_smolagents/{item_id}`).
    - Sends multiple requests to a non-blocking endpoint (`/`) to verify that the application remains responsive while CPU-heavy tasks are in progress.

## Setup

1.  **Python Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies:**
    Install the necessary Python packages from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **PostgreSQL Database:**
    - Make sure you have a PostgreSQL server running.
    - Create a database (e.g., `testdb`).
    - The application connects using `postgresql+asyncpg://postgres:postgres@localhost:5432/testdb`. Update `DATABASE_URL` in `app.py` if your credentials or database name differ.

## Running the Application

1.  **Start the FastAPI server:**
    ```bash
    uvicorn app:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`. It will automatically create the necessary tables and sample data in the database on startup.

## Running the Benchmark

1.  **Ensure the FastAPI application is running.**
2.  **Execute the benchmark script:**
    ```bash
    python benchmark.py
    ```
