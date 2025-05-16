import asyncio
import httpx
import time
import json

# Configuration
BASE_URL = "http://127.0.0.1:8000"  # Assuming the FastAPI app runs on port 8000
CPU_HEAVY_BASE_ENDPOINT = f"{BASE_URL}/process_item_smolagents"
NUM_CPU_HEAVY_REQUESTS = 3
NON_BLOCKING_ENDPOINT = f"{BASE_URL}/"
NUM_NON_BLOCKING_REQUESTS = 5


async def fetch_url(client: httpx.AsyncClient, url: str, request_name: str):
    """Fetches a URL, prints time taken, and validates stream completion if applicable."""
    start_time = time.perf_counter()
    print(f"[{request_name}] Sending request to {url}...")
    stream_data_valid = None  # True: success, False: failure/error, None: not a stream or not determined

    try:
        if CPU_HEAVY_BASE_ENDPOINT in url:
            # Handle streaming response
            last_json_chunk = None
            stream_content_received = False
            async with client.stream("GET", url, timeout=60.0) as response:
                response.raise_for_status()
                print(
                    f"[{request_name}] Stream opened for {url}. Status: {response.status_code}"
                )
                async for line in response.aiter_lines():
                    if line.strip():
                        stream_content_received = True
                        # print(f"[{request_name}] Received stream line (length: {len(line)}).") # Optional: verbose
                        try:
                            last_json_chunk = json.loads(line)
                        except json.JSONDecodeError:
                            print(
                                f"[{request_name}] Warning: Could not decode JSON from line: {line[:100]}..."
                            )
                            # If a line isn't JSON, we're interested if the *last successfully parsed* line was success.
                elapsed_time = time.perf_counter() - start_time
                print(
                    f"[{request_name}] Stream closed for {url} in {elapsed_time:.4f} seconds. Status: {response.status_code}"
                )

            if last_json_chunk:
                if (
                    last_json_chunk.get("type") == "done"
                    and last_json_chunk.get("status") == "success"
                ):
                    stream_data_valid = True
                    print(
                        f"[{request_name}] Stream final JSON indicates success: {last_json_chunk}"
                    )
                else:
                    stream_data_valid = False
                    print(
                        f"[{request_name}] Stream final JSON does NOT indicate success or is malformed: {last_json_chunk}"
                    )
            elif (
                stream_content_received
            ):  # Content received, but no valid JSON (or last line wasn't valid JSON)
                stream_data_valid = False
                print(
                    f"[{request_name}] Stream had content but no valid final JSON line found or last line was not valid JSON."
                )
            else:  # No content received at all
                stream_data_valid = False
                print(f"[{request_name}] No content received from stream for {url}.")
            return elapsed_time, stream_data_valid
        else:
            # Handle non-streaming response
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            elapsed_time = time.perf_counter() - start_time
            print(
                f"[{request_name}] Received response from {url} in {elapsed_time:.4f} seconds. Status: {response.status_code}"
            )
            # print(f"[{request_name}] Response JSON: {response.json()}")
            return elapsed_time, None  # None indicates not a stream to validate
    except httpx.ReadTimeout:
        elapsed_time = time.perf_counter() - start_time
        print(f"[{request_name}] Timeout for {url} after {elapsed_time:.4f} seconds.")
        return elapsed_time, False if CPU_HEAVY_BASE_ENDPOINT in url else None
    except httpx.HTTPStatusError as e:
        elapsed_time = time.perf_counter() - start_time
        print(
            f"[{request_name}] HTTP error for {url}: {e.response.status_code} - {e.response.text} in {elapsed_time:.4f} seconds."
        )
        return elapsed_time, False if CPU_HEAVY_BASE_ENDPOINT in url else None
    except httpx.RequestError as e:
        elapsed_time = time.perf_counter() - start_time
        print(
            f"[{request_name}] Request error for {url}: {e} in {elapsed_time:.4f} seconds."
        )
        return elapsed_time, False if CPU_HEAVY_BASE_ENDPOINT in url else None


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

        cpu_heavy_task_results = []  # Stores (time, stream_success_status)
        non_blocking_times = []  # Stores times

        for i, result_item in enumerate(results):
            is_cpu_heavy_task = i < NUM_CPU_HEAVY_REQUESTS
            task_idx = (
                (i + 1) if is_cpu_heavy_task else (i - NUM_CPU_HEAVY_REQUESTS + 1)
            )
            task_type_name = "CPU-Heavy" if is_cpu_heavy_task else "Non-Blocking"
            task_full_name = f"{task_type_name}-{task_idx}"

            if isinstance(result_item, Exception):
                print(f"Task {task_full_name} resulted in an error: {result_item}")
                if is_cpu_heavy_task:
                    cpu_heavy_task_results.append(
                        {"time": None, "success": False, "id": task_idx}
                    )
            elif is_cpu_heavy_task:
                elapsed_time, stream_success = result_item
                cpu_heavy_task_results.append(
                    {"time": elapsed_time, "success": stream_success, "id": task_idx}
                )
            else:  # Non-blocking tasks
                elapsed_time, _ = result_item  # Second element is None for non-blocking
                if elapsed_time is not None:  # Ensure time is not None before appending
                    non_blocking_times.append(elapsed_time)

        all_cpu_streams_ok = True
        if NUM_CPU_HEAVY_REQUESTS > 0:  # Only evaluate if CPU heavy tasks were expected
            if (
                not cpu_heavy_task_results
            ):  # Should not happen if NUM_CPU_HEAVY_REQUESTS > 0 and tasks were created
                all_cpu_streams_ok = False
            for res in cpu_heavy_task_results:
                if not res["success"]:
                    all_cpu_streams_ok = False
                    break

        valid_cpu_heavy_times = [
            r["time"] for r in cpu_heavy_task_results if r["time"] is not None
        ]

        if cpu_heavy_task_results:
            print("CPU-heavy tasks results:")
            for res in cpu_heavy_task_results:
                time_str = (
                    f"{res['time']:.4f} seconds"
                    if res["time"] is not None
                    else "N/A (task error)"
                )
                status_str = (
                    "successful final JSON"
                    if res["success"]
                    else "FAILED or unsuccessful final JSON"
                )
                print(
                    f"  CPU-Heavy-{res['id']} completed in: {time_str}, stream status: {status_str}"
                )

            if valid_cpu_heavy_times:
                if (
                    len(valid_cpu_heavy_times) > 0
                ):  # Check to avoid division by zero if all failed
                    avg_cpu_heavy_time = sum(valid_cpu_heavy_times) / len(
                        valid_cpu_heavy_times
                    )
                    max_cpu_heavy_time = max(valid_cpu_heavy_times)
                    min_cpu_heavy_time = min(valid_cpu_heavy_times)
                    print(
                        f"  Average CPU-heavy task time (for completed): {avg_cpu_heavy_time:.4f} seconds"
                    )
                    print(
                        f"  Min CPU-heavy task time (for completed):     {min_cpu_heavy_time:.4f} seconds"
                    )
                    print(
                        f"  Max CPU-heavy task time (for completed):     {max_cpu_heavy_time:.4f} seconds"
                    )
            if (
                NUM_CPU_HEAVY_REQUESTS > 0
            ):  # Only print this part if CPU tasks were expected
                if all_cpu_streams_ok:
                    print("  All CPU-heavy streams ended with a successful final JSON.")
                else:
                    print(
                        "  WARNING: One or more CPU-heavy streams did NOT end with a successful final JSON or failed."
                    )

        max_non_blocking_time = 0.0
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
        elif NUM_NON_BLOCKING_REQUESTS > 0:
            print("No non-blocking tasks completed successfully.")

        print("-" * 30)
        non_blocking_check_passed = (NUM_NON_BLOCKING_REQUESTS == 0) or (
            non_blocking_times and max_non_blocking_time < 1.0
        )
        cpu_heavy_check_passed = (NUM_CPU_HEAVY_REQUESTS == 0) or all_cpu_streams_ok

        if non_blocking_check_passed and cpu_heavy_check_passed:
            print(
                "Benchmark PASSED: Non-blocking requests were processed quickly (if any) "
                + "AND all CPU-heavy streams reported success (if any)."
            )
        else:
            reasons = []
            if NUM_NON_BLOCKING_REQUESTS > 0 and not non_blocking_check_passed:
                if not non_blocking_times:
                    reasons.append(
                        "No non-blocking tasks completed successfully to analyze timing."
                    )
                else:
                    reasons.append(
                        f"Non-blocking requests may have been delayed (max time: {max_non_blocking_time:.4f}s)."
                    )

            if NUM_CPU_HEAVY_REQUESTS > 0 and not cpu_heavy_check_passed:
                reasons.append(
                    "One or more CPU-heavy streams did not report success in their final JSON or failed."
                )

            status_text = (
                "FAILED"
                if (NUM_NON_BLOCKING_REQUESTS > 0 and not non_blocking_check_passed)
                or (NUM_CPU_HEAVY_REQUESTS > 0 and not cpu_heavy_check_passed)
                else "WARNING"
            )

            if (
                not reasons
            ):  # Should not happen if we are in this else block with positive requests
                reasons.append(
                    "General benchmark criteria not met. Review logs and times."
                )

            print(f"Benchmark {status_text}: {'; '.join(reasons)}")


if __name__ == "__main__":
    asyncio.run(main())
