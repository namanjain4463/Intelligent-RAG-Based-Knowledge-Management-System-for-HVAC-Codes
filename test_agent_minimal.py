"""
Minimal Agent Test - Tests just 5 key queries to verify functionality  
"""
import time
from agent import query_agent

print("\n" + "="*80)
print("HVAC AGENT - MINIMAL TEST (5 Queries)")
print("="*80 + "\n")

test_queries = [
    ("SIMPLE", "What is Section 303.3?"),
    ("MEDIUM", "Where are furnaces prohibited?"),
    ("MEDIUM", "What clearances does a furnace need?"),
    ("HARD", "What equipment is prohibited in bedrooms?"),
    ("COMPLEX", "Can I install a furnace in a bedroom closet?"),
]

results = []

for difficulty, query in test_queries:
    print(f"\n[{difficulty}] Testing: {query}")
    print("-"*80)
    
    start = time.time()
    try:
        response = query_agent(query)
        duration = time.time() - start
        
        # Check for success
        has_content = len(response) > 50
        no_error = "error" not in response.lower()
        has_section = "section" in response.lower()
        
        success = has_content and no_error
        
        print(f"\n[{'PASS' if success else 'FAIL'}] Duration: {duration:.2f}s")
        print(f"Response length: {len(response)} chars")
        print(f"Has section ref: {has_section}")
        print(f"\nResponse preview:")
        print(response[:300] + "..." if len(response) > 300 else response)
        
        results.append({
            "difficulty": difficulty,
            "query": query,
            "success": success,
            "duration": duration,
            "has_section": has_section
        })
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        results.append({
            "difficulty": difficulty,
            "query": query,
            "success": False,
            "duration": time.time() - start,
            "has_section": False
        })
    
    time.sleep(2)  # Rate limiting

# Summary
print("\n\n" + "="*80)
print("SUMMARY")
print("="*80)

total = len(results)
passed = sum(1 for r in results if r["success"])

print(f"\nTotal: {total}")
print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
print(f"Avg Duration: {sum(r['duration'] for r in results)/total:.2f}s")

print(f"\nDetailed Results:")
for i, r in enumerate(results, 1):
    status = "PASS" if r["success"] else "FAIL"
    print(f"  {i}. [{status}] {r['difficulty']:10s} {r['duration']:5.2f}s - {r['query'][:50]}")

print("\n" + "="*80)
