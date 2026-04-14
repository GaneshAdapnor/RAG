"""
End-to-end example: upload a document and ask a question.

This script demonstrates the full RAG pipeline via the HTTP API:
    1. Create a sample TXT document
    2. Upload it via POST /upload
    3. Poll GET /upload/status/{doc_id} until processing completes
    4. Query via POST /query
    5. Display the answer and source attribution

Run:
    python examples/sample_query.py

Requirements:
    pip install requests
    The API server must be running: uvicorn app.main:app --reload
"""

import sys
import time
import textwrap

try:
    import requests
except ImportError:
    print("ERROR: Install requests: pip install requests")
    sys.exit(1)

BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Sample document — a short fictional research summary
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENT = """
Quantum Battery Research: Annual Report 2024

Executive Summary
=================
The QuantumVolt research team achieved a breakthrough in solid-state battery technology
during Q3 2024. A novel lithium-ceramic composite electrolyte demonstrated an energy
density of 450 Wh/kg — approximately 2.5 times greater than conventional lithium-ion
batteries. Charge time from 0 to 80% was reduced to 8 minutes under laboratory conditions.

Key Findings
============
1. Energy Density: The new composite achieved 450 Wh/kg at room temperature,
   compared to the industry standard of 180 Wh/kg for commercial lithium-ion cells.

2. Cycle Life: The prototype maintained 92% capacity after 1,200 full charge-discharge
   cycles, exceeding the 80% retention threshold required for commercial viability.

3. Safety Profile: Unlike liquid electrolyte batteries, the solid-state design
   eliminates thermal runaway risk. No fire incidents occurred across 3,400 test cycles.

4. Manufacturing Cost: Current production cost is estimated at $185 per kWh,
   with projected reduction to $72 per kWh at scale. Commercial lithium-ion averages $110/kWh.

Challenges and Next Steps
=========================
The primary remaining challenge is interface resistance between the ceramic electrolyte
and electrode materials at temperatures below -10°C. Performance degrades 18% in cold
conditions. The team plans to address this through surface coating techniques in 2025.

Commercial pilot production is scheduled for Q2 2025, targeting EV manufacturers
and grid storage applications. The projected market introduction is 2027.

Team and Investment
===================
The project involved 47 researchers across materials science, electrochemistry,
and manufacturing engineering departments. Total R&D investment in 2024 was $23.4 million,
funded by a combination of government grants (60%) and private investment (40%).
""".strip()


def upload_document(content: str, filename: str = "quantum_battery_report.txt") -> str:
    """Upload a text document and return the doc_id."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Uploading '{filename}'...")
    print(f"{'='*60}")

    response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
        timeout=30,
    )

    if response.status_code != 202:
        print(f"Upload failed ({response.status_code}): {response.text}")
        sys.exit(1)

    data = response.json()
    doc_id = data["doc_id"]
    print(f"Accepted: doc_id={doc_id}")
    print(f"Status:   {data['status']}")
    print(f"Message:  {data['message']}")
    return doc_id


def wait_for_processing(doc_id: str, max_wait_seconds: int = 120) -> dict:
    """Poll processing status until completed or failed."""
    print(f"\n{'='*60}")
    print("STEP 2: Waiting for background processing...")
    print(f"{'='*60}")

    start = time.time()
    poll_interval = 2  # seconds

    while time.time() - start < max_wait_seconds:
        response = requests.get(f"{BASE_URL}/upload/status/{doc_id}", timeout=10)
        if response.status_code != 200:
            print(f"Status check failed: {response.text}")
            sys.exit(1)

        status_data = response.json()
        status = status_data["status"]
        elapsed = time.time() - start

        print(f"  [{elapsed:5.1f}s] status={status}", end="")
        if status == "completed":
            print(f" | chunks={status_data['chunk_count']}")
            return status_data
        elif status == "failed":
            print(f" | error={status_data['error']}")
            sys.exit(1)
        else:
            print()

        time.sleep(poll_interval)

    print(f"Timed out after {max_wait_seconds}s waiting for processing.")
    sys.exit(1)


def ask_question(query: str, doc_ids: list = None) -> dict:
    """Submit a query and return the full response."""
    print(f"\n{'='*60}")
    print("STEP 3: Querying the RAG system...")
    print(f"{'='*60}")
    print(f"Query: {query}")

    payload = {"query": query, "top_k": 5}
    if doc_ids:
        payload["doc_ids"] = doc_ids

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        print(f"Query failed ({response.status_code}): {response.text}")
        sys.exit(1)

    return response.json()


def display_result(result: dict) -> None:
    """Pretty-print the query result."""
    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")

    print(f"\n[ANSWER]\n")
    # Wrap long lines for readability
    for line in result["answer"].splitlines():
        print(textwrap.fill(line, width=70) if line else "")

    print(f"\n[LATENCY]")
    print(f"  Retrieval (embed + FAISS):  {result['retrieval_latency_ms']:.1f} ms")
    print(f"  Generation (OpenAI API):    {result['generation_latency_ms']:.1f} ms")
    total = result['retrieval_latency_ms'] + result['generation_latency_ms']
    print(f"  Total:                      {total:.1f} ms")

    print(f"\n[SOURCES] ({len(result['sources'])} chunks retrieved)")
    for i, src in enumerate(result["sources"], 1):
        print(f"\n  Source {i}: {src['filename']}", end="")
        if src.get("page"):
            print(f", page {src['page']}", end="")
        print(f" | similarity={src['score']:.4f}")
        # Show first 200 chars of the chunk
        snippet = src["text"][:200].replace("\n", " ")
        print(f"  '{snippet}...'")


def check_health() -> None:
    """Verify the API is running."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        data = r.json()
        print(f"Health: {data['status']} | model={data['embedding_model']} | "
              f"chunks={data['indexed_chunks']} | docs={data['indexed_documents']}")
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to API at {BASE_URL}")
        print("Make sure the server is running:")
        print("  uvicorn app.main:app --reload")
        sys.exit(1)


if __name__ == "__main__":
    print("Production RAG QA System — End-to-End Test")
    print("=" * 60)

    # 0. Health check
    check_health()

    # 1. Upload document
    doc_id = upload_document(SAMPLE_DOCUMENT)

    # 2. Wait for ingestion
    status = wait_for_processing(doc_id)
    print(f"\nDocument ready: {status['chunk_count']} chunks indexed.")

    # 3 & 4. Run example queries
    queries = [
        "What energy density did the new battery achieve?",
        "What are the main challenges with the solid-state battery?",
        "How much did the research cost and who funded it?",
        "What is the projected commercial launch date?",
    ]

    for query in queries:
        result = ask_question(query, doc_ids=[doc_id])
        display_result(result)
        print()
        time.sleep(1)  # Brief pause between queries (rate limiting)
