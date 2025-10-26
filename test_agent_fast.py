"""
Fast Agent Test - Summary only, no verbose output
Tests key query types to identify what works and what doesn't
"""
import time
import logging
from agent import executor

# Suppress verbose logging
logging.getLogger("agent").setLevel(logging.ERROR)
logging.getLogger("cypher_generator").setLevel(logging.ERROR)
logging.getLogger("llm").setLevel(logging.ERROR)

def test_query(query):
    """Test a single query - returns success/failure"""
    print(f"Testing: {query[:70]:<70} ... ", end="", flush=True)
    
    start = time.time()
    try:
        result = executor.invoke({"input": query})
        duration = time.time() - start
        
        response = result.get("output", "")
        
        # Success criteria
        has_content = len(response) > 50
        no_error = "error" not in response.lower() and "failed" not in response.lower()
        has_section_ref = "section" in response.lower() or "code" in response.lower()
        
        success = has_content and no_error
        
        print(f"[{'PASS' if success else 'FAIL'}] {duration:.1f}s")
        
        return {
            "query": query,
            "success": success,
            "duration": duration,
            "response_length": len(response),
            "has_section_ref": has_section_ref
        }
    except Exception as e:
        duration = time.time() - start
        print(f"[ERROR] {duration:.1f}s - {str(e)[:40]}")
        return {
            "query": query,
            "success": False,
            "duration": duration,
            "response_length": 0,
            "has_section_ref": False
        }

# Test Suite
print("\n" + "="*100)
print("HVAC AGENT TEST SUITE - Fast Mode (Summary Only)")
print("="*100 + "\n")

all_tests = []

# === LEVEL 1: SIMPLE (Single Entity Lookups) ===
print("\n[LEVEL 1: SIMPLE] Single entity lookups, basic facts")
print("-"*100)

simple_tests = [
    "What is Section 303.3?",
    "List all equipment types",
    "What locations are in the database?",
    "What is Section 301?",
]

for query in simple_tests:
    all_tests.append(test_query(query))
    time.sleep(1)

# === LEVEL 2: MEDIUM (Simple Relationships) ===
print("\n[LEVEL 2: MEDIUM] Simple relationships, single-hop queries")
print("-"*100)

medium_tests = [
    "Where are furnaces prohibited?",
    "What clearances does a furnace need?",
    "What safety devices does a boiler require?",
    "Where can I install a water heater?",
    "What standards must equipment comply with?",
]

for query in medium_tests:
    all_tests.append(test_query(query))
    time.sleep(1)

# === LEVEL 3: HARD (Multiple Hops, Filtering) ===
print("\n[LEVEL 3: HARD] Multiple hops, filtering, aggregation")
print("-"*100)

hard_tests = [
    "What equipment is prohibited in bedrooms?",
    "What are the clearance requirements for combustible materials?",
    "What sections are in Chapter 3?",
    "Show me TABLE 305.4",
    "What equipment requires relief valves?",
]

for query in hard_tests:
    all_tests.append(test_query(query))
    time.sleep(1)

# === LEVEL 4: COMPLEX (Multi-step Reasoning) ===
print("\n[LEVEL 4: COMPLEX] Multi-step reasoning, combinations")
print("-"*100)

complex_tests = [
    "Can I install a furnace in a bedroom closet?",
    "What are all the requirements for installing a boiler in a mechanical room?",
    "What equipment requires both relief valves and disconnects?",
    "What's the difference between Section 303.3 and Section 303.8?",
]

for query in complex_tests:
    all_tests.append(test_query(query))
    time.sleep(1)

# === LEVEL 5: SUPER COMPLEX (Domain Reasoning, Real-world) ===
print("\n[LEVEL 5: SUPER COMPLEX] Domain reasoning, real-world scenarios")
print("-"*100)

super_complex_tests = [
    "I want to install a gas furnace in my garage near wooden walls. What code requirements must I follow?",
    "What are the ventilation requirements for a mechanical room with a boiler?",
    "Compare installation requirements for furnaces vs boilers",
    "What are the exceptions to Section 303.3?",
]

for query in super_complex_tests:
    all_tests.append(test_query(query))
    time.sleep(1)

# === SUMMARY REPORT ===
print("\n\n" + "="*100)
print("SUMMARY REPORT")
print("="*100 + "\n")

total = len(all_tests)
passed = sum(1 for t in all_tests if t["success"])
failed = total - passed
avg_duration = sum(t["duration"] for t in all_tests) / total

print(f"Overall Statistics:")
print(f"  Total Tests:      {total}")
print(f"  Passed:           {passed} ({passed/total*100:.1f}%)")
print(f"  Failed:           {failed} ({failed/total*100:.1f}%)")
print(f"  Avg Duration:     {avg_duration:.2f}s")

# By difficulty
print(f"\nSuccess Rate by Difficulty:")
levels = [
    ("SIMPLE", 0, 4),
    ("MEDIUM", 4, 9),
    ("HARD", 9, 14),
    ("COMPLEX", 14, 18),
    ("SUPER COMPLEX", 18, 22),
]

for level_name, start_idx, end_idx in levels:
    level_tests = all_tests[start_idx:end_idx]
    level_passed = sum(1 for t in level_tests if t["success"])
    level_total = len(level_tests)
    if level_total > 0:
        print(f"  {level_name:15s}: {level_passed}/{level_total} ({level_passed/level_total*100:.1f}%)")

# Failed tests
failed_tests = [t for t in all_tests if not t["success"]]
if failed_tests:
    print(f"\nFailed Tests ({len(failed_tests)}):")
    for i, t in enumerate(failed_tests, 1):
        print(f"  {i}. {t['query'][:80]}")

# Section reference coverage
with_section_refs = sum(1 for t in all_tests if t["success"] and t["has_section_ref"])
print(f"\nSection Reference Coverage:")
print(f"  Tests with section refs: {with_section_refs}/{passed} ({with_section_refs/passed*100 if passed > 0 else 0:.1f}%)")

# Performance
fastest = min(all_tests, key=lambda t: t["duration"])
slowest = max(all_tests, key=lambda t: t["duration"])
print(f"\nPerformance:")
print(f"  Fastest: {fastest['duration']:.2f}s - {fastest['query'][:60]}")
print(f"  Slowest: {slowest['duration']:.2f}s - {slowest['query'][:60]}")

print("\n" + "="*100)
print("Test Complete!")
print("="*100)
