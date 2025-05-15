import asyncio
import httpx
import time

# Configuration
BASE_URL = "http://127.0.0.1:8000"  # Assuming the FastAPI app runs on port 8000
CPU_HEAVY_BASE_ENDPOINT = f"{BASE_URL}/process_item_smolagents"
NUM_CPU_HEAVY_REQUESTS = 3
NON_BLOCKING_ENDPOINT = f"{BASE_URL}/"
NUM_NON_BLOCKING_REQUESTS = 5


async def fetch_url(client: httpx.AsyncClient, url: str, request_name: str):
    """Fetches a URL and prints the time taken."""
    start_time = time.perf_counter()
    print(f"[{request_name}] Sending request to {url}...")
    try:
        response = await client.get(
            url, timeout=30.0
        )  # Increased timeout for potentially long op
        response.raise_for_status()  # Raise an exception for bad status codes
        elapsed_time = time.perf_counter() - start_time
        print(
            f"[{request_name}] Received response from {url} in {elapsed_time:.4f} seconds. Status: {response.status_code}"
        )
        # print(f"[{request_name}] Response JSON: {response.json()}")
        return elapsed_time
    except httpx.ReadTimeout:
        elapsed_time = time.perf_counter() - start_time
        print(f"[{request_name}] Timeout for {url} after {elapsed_time:.4f} seconds.")
        return elapsed_time
    except httpx.HTTPStatusError as e:
        elapsed_time = time.perf_counter() - start_time
        print(
            f"[{request_name}] HTTP error for {url}: {e.response.status_code} - {e.response.text} in {elapsed_time:.4f} seconds."
        )
        return elapsed_time
    except httpx.RequestError as e:
        elapsed_time = time.perf_counter() - start_time
        print(
            f"[{request_name}] Request error for {url}: {e} in {elapsed_time:.4f} seconds."
        )
        return elapsed_time


async def main():
    """Runs the benchmark."""
    print("Starting benchmark...")
    print(
        f"Targeting {NUM_CPU_HEAVY_REQUESTS} CPU-heavy endpoints: {CPU_HEAVY_BASE_ENDPOINT}/{{item_id}}"
    )
    print(
        f"Targeting non-blocking endpoint: {NON_BLOCKING_ENDPOINT} ({NUM_NON_BLOCKING_REQUESTS} times)"
    )
    print("-" * 30)

    async with httpx.AsyncClient() as client:
        tasks = []

        # Start the CPU-heavy tasks
        print(f"Dispatching {NUM_CPU_HEAVY_REQUESTS} CPU-heavy tasks...")
        for i in range(NUM_CPU_HEAVY_REQUESTS):
            item_id = i + 1
            endpoint = f"{CPU_HEAVY_BASE_ENDPOINT}/{item_id}"
            cpu_heavy_task = asyncio.create_task(
                fetch_url(client, endpoint, f"CPU-Heavy-{item_id}")
            )
            tasks.append(cpu_heavy_task)

        # Give a very small delay to ensure the CPU-heavy requests are likely sent first
        await asyncio.sleep(0.1)

        # Start non-blocking tasks
        print(f"Dispatching {NUM_NON_BLOCKING_REQUESTS} non-blocking tasks...")
        for i in range(NUM_NON_BLOCKING_REQUESTS):
            task = asyncio.create_task(
                fetch_url(client, NON_BLOCKING_ENDPOINT, f"Non-Blocking-{i+1}")
            )
            tasks.append(task)

        # Wait for all tasks to complete
        print("-" * 30)
        print("Waiting for all tasks to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print("-" * 30)

        cpu_heavy_times = []
        non_blocking_times = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if i < NUM_CPU_HEAVY_REQUESTS:
                    task_name = f"CPU-Heavy-{i+1}"
                else:
                    task_name = f"Non-Blocking-{i - NUM_CPU_HEAVY_REQUESTS + 1}"
                print(f"Task {task_name} resulted in an error: {result}")
            elif i < NUM_CPU_HEAVY_REQUESTS:  # CPU-heavy tasks
                cpu_heavy_times.append(result)
            else:  # Non-blocking tasks
                non_blocking_times.append(result)

        if cpu_heavy_times:
            print("CPU-heavy tasks completed:")
            for i, t in enumerate(cpu_heavy_times):
                print(f"  CPU-Heavy-{i+1} task completed in: {t:.4f} seconds")
            if len(cpu_heavy_times) > 1:
                avg_cpu_heavy_time = sum(cpu_heavy_times) / len(cpu_heavy_times)
                max_cpu_heavy_time = max(cpu_heavy_times)
                min_cpu_heavy_time = min(cpu_heavy_times)
                print(f"  Average CPU-heavy time: {avg_cpu_heavy_time:.4f} seconds")
                print(f"  Min CPU-heavy time:     {min_cpu_heavy_time:.4f} seconds")
                print(f"  Max CPU-heavy time:     {max_cpu_heavy_time:.4f} seconds")

        if non_blocking_times:
            avg_non_blocking_time = sum(non_blocking_times) / len(non_blocking_times)
            max_non_blocking_time = max(non_blocking_times)
            min_non_blocking_time = min(non_blocking_times)
            print("Non-blocking tasks completed with:")
            print(f"  Average time: {avg_non_blocking_time:.4f} seconds")
            print(f"  Min time:     {min_non_blocking_time:.4f} seconds")
            print(f"  Max time:     {max_non_blocking_time:.4f} seconds")
            print(
                f"  Individual times: {[float(f'{t:.4f}') for t in non_blocking_times]}"
            )

        print("-" * 30)
        if (
            non_blocking_times and max_non_blocking_time < 1.0
        ):  # Assuming non-blocking should be very fast
            print(
                "Benchmark PASSED: Non-blocking requests were processed quickly while the CPU-heavy task was running."
            )
        elif not non_blocking_times:
            print(
                "Benchmark WARNING: No non-blocking tasks completed successfully to analyze."
            )
        else:
            print(
                "Benchmark FAILED or WARNING: Non-blocking requests may have been delayed. Review times."
            )


if __name__ == "__main__":
    # Ensure there are items with IDs 1, 2, 3 (or up to NUM_CPU_HEAVY_REQUESTS)
    # in your database for the CPU_HEAVY_ENDPOINTS.
    # The app.py already creates sample items (Sample Item 1, Sample Item 2, Sample Item 3),
    # so item_ids 1, 2, and 3 should exist if NUM_CPU_HEAVY_REQUESTS is 3 or less.
    asyncio.run(main())
