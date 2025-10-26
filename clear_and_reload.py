"""
CLEAR DATABASE AND RELOAD WITH SECTION CONTENT
"""
from graph import graph

print("=" * 60)
print("CLEARING NEO4J DATABASE")
print("=" * 60)

# Delete all nodes and relationships
result = graph.query("MATCH (n) DETACH DELETE n")
print("✓ All nodes and relationships deleted")

# Verify empty
stats = graph.query("""
    MATCH (n)
    RETURN count(n) as total
""")
print(f"✓ Database is empty: {stats[0]['total']} nodes remaining")

print("\n" + "=" * 60)
print("NOW RUN: python extract_and_load.py")
print("=" * 60)
