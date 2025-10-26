"""
PDF Extraction and Neo4j Database Population (Phase 2A - Domain-Driven)
RUN THIS FIRST: python extract_and_load.py

This script:
1. Extracts text from HVAC-Codes.pdf
2. Builds hierarchical section structure (901 → 901.1 → 901.1.1)
3. Extracts domain entities (Equipment, Location, Material, Standard, SafetyDevice)
4. Extracts domain relationships using NLP (PROHIBITED_IN, REQUIRES_CLEARANCE, etc.)
5. Creates all nodes and relationships in Neo4j with section links
6. Extracts and stores tables with proper Table/TableRow structure
7. Generates embeddings for semantic search
"""
import fitz  # PyMuPDF
import re
from graph import graph
from llm import embeddings
from utils import build_materialized_path, clean_text, chunk_text
from config import Config
from tqdm import tqdm
from vocabulary import entity_normalizer, EntityNormalizer
from relationship_extractor import RelationshipExtractor
from table_extractor import TableExtractor

def relationship_exists(source: str, target: str, rel_type: str) -> bool:
    """Check if a relationship already exists between two nodes"""
    query = f"""
    MATCH (a {{name: $source}})-[r:{rel_type}]->(b {{name: $target}})
    RETURN count(r) as count
    """
    result = graph.query(query, {"source": source, "target": target})
    return result[0]['count'] > 0 if result else False

def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from PDF using PyMuPDF"""
    print(f"\n[1/7] Extracting text from {pdf_path}...")
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in tqdm(range(len(doc)), desc="Pages"):
        page = doc[page_num]
        text += page.get_text()
    
    doc.close()
    cleaned = clean_text(text)
    print(f"✓ Extracted {len(cleaned)} characters")
    return cleaned

def extract_sections(text: str):
    """Extract all sections with hierarchy using regex - captures ALL nested subsections"""
    print("\n[2/7] Extracting sections...")
    
    # ENHANCED PATTERN: Captures both "Section XXX" headers AND standalone subsection numbers
    # Pattern 1: "Section XXX Title" or "SECTION XXX Title"
    # Pattern 2: Standalone subsection numbers like "301.1", "301.8.1.2.3" at line start
    
    sections = []
    seen_numbers = set()  # Track to avoid duplicates
    
    # Pattern 1: Explicit "Section XXX Title" format
    section_header_pattern = re.compile(
        r'(?:Section|SECTION)\s+(\d{1,4}(?:\.\d+)*)\s+([^\n]+)',
        re.IGNORECASE
    )
    
    # Pattern 2: Subsection numbers at line start (e.g., "301.1 Title", "303.3.1 Title")
    # This captures nested subsections that don't have "Section" prefix
    subsection_pattern = re.compile(
        r'^\s*(\d{1,4}\.\d+(?:\.\d+)*)\s+([^\n]+?)(?:\n|$)',
        re.MULTILINE
    )
    
    # First, extract explicit "Section XXX" headers
    for match in section_header_pattern.finditer(text):
        number = match.group(1)
        title = match.group(2).strip()
        
        # Skip if already found
        if number in seen_numbers:
            continue
        
        # Filter out malformed entries
        if len(title) < 3:
            continue
        
        if title and title[0].islower():
            continue
        
        if title.startswith(('of the', 'and ', 'or ', 'for ', 'in accordance')):
            continue
        
        seen_numbers.add(number)
        
        # Build materialized path
        try:
            path_info = build_materialized_path(number)
            sections.append({
                "number": number,
                "path": path_info["path"],
                "level": path_info["level"],
                "title": title
            })
        except (ValueError, IndexError, AttributeError) as e:
            print(f"⚠️  Warning: Could not build path for section {number}: {e}")
            continue
    
    # Second, extract standalone subsection numbers (301.1, 301.8.1, etc.)
    for match in subsection_pattern.finditer(text):
        number = match.group(1)
        title = match.group(2).strip()
        
        # Skip if already found
        if number in seen_numbers:
            continue
        
        # Filter out malformed entries
        # Must have at least one dot (is a subsection)
        if '.' not in number:
            continue
        
        # CRITICAL FIX: Skip table measurements (numbers < 1.0)
        # Examples: 0.028 (gauge thickness), 0.013 (dimensions), 0.25 (ACH values)
        # Valid section numbers start at 1.0 or higher (301.1, 803.10.4, etc.)
        try:
            first_part = int(number.split('.')[0])
            if first_part < 1:
                continue  # Skip decimal table data (0.028, 0.013, etc.)
            
            # CRITICAL FIX 2: Skip table rows (single/double digit section numbers)
            # Examples: 1.0 (table row), 1.5 (table row), 10.6 (table row)
            # Valid HVAC section numbers are 3+ digits: 301, 303, 803, 1001, 1002, etc.
            if first_part < 100:
                continue  # Skip table rows (1.0, 1.5, 2.0, 10.6, etc.)
                
        except (ValueError, IndexError):
            continue  # Skip if parsing fails
        
        # Additional filter: Title should not be purely numeric (table values)
        # Examples: "8,50", "22,400", "12,0"
        if re.match(r'^[\d,\.\s]+$', title):
            continue  # Skip numeric-only titles (table data)
        
        # Title should be reasonable length
        if len(title) < 3 or len(title) > 200:
            continue
        
        # Skip if title starts with lowercase (likely part of sentence)
        if title and title[0].islower():
            continue
        
        # Skip common false positives
        if title.startswith(('of the', 'and ', 'or ', 'for ', 'in accordance', 'shall ', 'must ', 'where ')):
            continue
        
        # Skip if title ends with colon (likely incomplete)
        if title.endswith(':'):
            continue
        
        # Skip if number looks invalid (e.g., 1.1.1.1.1.1.1)
        if number.count('.') > 5:  # Max 5 levels deep
            continue
        
        seen_numbers.add(number)
        
        # Build materialized path
        try:
            path_info = build_materialized_path(number)
            sections.append({
                "number": number,
                "path": path_info["path"],
                "level": path_info["level"],
                "title": title
            })
        except (ValueError, IndexError, AttributeError) as e:
            # Skip if path building fails
            continue
    
    # Sort sections by path for proper hierarchy
    sections.sort(key=lambda x: x['path'])
    
    print(f"✓ Found {len(sections)} sections")
    
    # Show sample of what was found
    if sections:
        print(f"  Sample sections:")
        for s in sections[:10]:
            print(f"    {s['number']}: {s['title'][:60]}...")
        if len(sections) > 10:
            print(f"    ... and {len(sections) - 10} more")
    
    return sections

def create_document_node():
    """Create root Document node"""
    print("\n[3/7] Creating Document node...")
    graph.query("""
    MERGE (d:Document {name: "HVAC Codes 2021", version: "2021", date: "2021-01-01"})
    """)
    print("✓ Document node created")

def create_chapter_nodes(sections):
    """Create Chapter nodes"""
    print("\n[4/7] Creating Chapter nodes...")
    chapters = set()
    
    for section in sections:
        # Get first digit(s) as chapter number
        if len(section["number"]) <= 3:
            chapters.add(section["number"][0])
    
    for chapter_num in tqdm(chapters, desc="Chapters"):
        graph.query("""
        MERGE (c:Chapter {number: $num, title: $title})
        MERGE (d:Document {name: "HVAC Codes 2021"})-[:CONTAINS]->(c)
        """, {"num": chapter_num, "title": f"Chapter {chapter_num}"})
    
    print(f"✓ Created {len(chapters)} chapters")

def create_section_nodes(sections):
    """Create Section nodes with materialized paths"""
    print("\n[5/7] Creating Section nodes with hierarchy...")
    
    for section in tqdm(sections, desc="Sections"):
        # Create section node
        graph.query("""
        MERGE (s:Section {
            number: $number,
            path: $path,
            level: $level,
            title: $title
        })
        """, section)
        
        # Create CONTAINS relationships
        if section["level"] == 1:
            # Top-level section: Chapter contains it
            chapter_num = section["number"][0]
            graph.query("""
            MATCH (c:Chapter {number: $chapter})
            MATCH (s:Section {number: $section})
            MERGE (c)-[:CONTAINS]->(s)
            """, {"chapter": chapter_num, "section": section["number"]})
        else:
            # Nested section: Parent section contains it
            parts = section["number"].split('.')
            parent_num = '.'.join(parts[:-1])
            graph.query("""
            MATCH (parent:Section {number: $parent})
            MATCH (child:Section {number: $child})
            MERGE (parent)-[:CONTAINS]->(child)
            """, {"parent": parent_num, "child": section["number"]})
    
    print(f"✓ Created {len(sections)} sections")

def create_domain_entity_nodes():
    """Create Equipment, Location, Material, Standard, and SafetyDevice nodes with synonyms"""
    print("\n[6/7] Creating domain entity nodes...")
    
    normalizer = EntityNormalizer()
    
    # Equipment nodes
    for canonical, synonyms in tqdm(normalizer.EQUIPMENT_CANONICAL.items(), desc="Equipment"):
        graph.query("""
        MERGE (e:Equipment {name: $name})
        SET e.synonyms = $synonyms, e.type = "HVAC Equipment"
        """, {"name": canonical, "synonyms": synonyms})
    
    # Location nodes
    for canonical, synonyms in tqdm(normalizer.LOCATION_CANONICAL.items(), desc="Locations"):
        graph.query("""
        MERGE (l:Location {name: $name})
        SET l.synonyms = $synonyms, l.type = "Building Location"
        """, {"name": canonical, "synonyms": synonyms})
    
    # Material nodes
    for canonical, synonyms in tqdm(normalizer.MATERIAL_CANONICAL.items(), desc="Materials"):
        graph.query("""
        MERGE (m:Material {name: $name})
        SET m.synonyms = $synonyms, m.type = "Construction Material"
        """, {"name": canonical, "synonyms": synonyms})
    
    # Standard nodes
    for canonical, synonyms in tqdm(normalizer.STANDARD_CANONICAL.items(), desc="Standards"):
        graph.query("""
        MERGE (s:Standard {name: $name})
        SET s.synonyms = $synonyms, s.type = "Code/Standard"
        """, {"name": canonical, "synonyms": synonyms})
    
    # SafetyDevice nodes
    for canonical, synonyms in tqdm(normalizer.SAFETY_DEVICE_CANONICAL.items(), desc="SafetyDevices"):
        graph.query("""
        MERGE (sd:SafetyDevice {name: $name})
        SET sd.synonyms = $synonyms, sd.type = "Safety Equipment"
        """, {"name": canonical, "synonyms": synonyms})
    
    equipment_count = len(normalizer.EQUIPMENT_CANONICAL)
    location_count = len(normalizer.LOCATION_CANONICAL)
    material_count = len(normalizer.MATERIAL_CANONICAL)
    standard_count = len(normalizer.STANDARD_CANONICAL)
    safety_device_count = len(normalizer.SAFETY_DEVICE_CANONICAL)
    
    print(f"✓ Created {equipment_count} Equipment, {location_count} Locations, {material_count} Materials, {standard_count} Standards, {safety_device_count} SafetyDevices")

def extract_section_content(text: str, section_number: str, sections: list) -> str:
    """Extract the text content for a specific section or subsection."""
    # Pattern 1: Try "Section XXX Title" format first
    section_pattern = re.compile(
        rf'(?:Section|SECTION)\s+{re.escape(section_number)}\s+([^\n]+)',
        re.IGNORECASE
    )
    
    match = section_pattern.search(text)
    
    # Pattern 2: If not found, try subsection number at line start (e.g., "303.3 Title")
    if not match:
        subsection_pattern = re.compile(
            rf'^\s*{re.escape(section_number)}\s+([^\n]+)',
            re.MULTILINE
        )
        match = subsection_pattern.search(text)
    
    if not match:
        return ""
    
    start_pos = match.start()
    
    # Find next section to determine end position
    # Sort sections by their number to find the next one
    section_nums = [s['number'] for s in sections]
    try:
        idx = section_nums.index(section_number)
        if idx + 1 < len(section_nums):
            next_section = section_nums[idx + 1]
            
            # Try both patterns for next section
            next_pattern1 = re.compile(
                rf'(?:Section|SECTION)\s+{re.escape(next_section)}\s+',
                re.IGNORECASE
            )
            next_pattern2 = re.compile(
                rf'^\s*{re.escape(next_section)}\s+[A-Z]',
                re.MULTILINE
            )
            
            next_match = next_pattern1.search(text, start_pos + 1)
            if not next_match:
                next_match = next_pattern2.search(text, start_pos + 1)
                
            if next_match:
                end_pos = next_match.start()
                return text[start_pos:end_pos]
    except ValueError:
        pass
    
    # If no next section found, take next 2000 characters as reasonable chunk
    return text[start_pos:start_pos + 2000]

def create_domain_relationships(text, sections):
    """Extract and create domain relationships using NLP"""
    print("\n[7/7] Extracting domain relationships...")
    
    extractor = RelationshipExtractor()
    relationship_stats = {
        'PROHIBITED_IN': 0,
        'REQUIRES_CLEARANCE': 0,
        'MUST_COMPLY_WITH': 0,
        'PERMITTED_IN': 0,
        'REQUIRES_DEVICE': 0,
        'MADE_OF': 0
    }
    
    for section in tqdm(sections, desc="Analyzing sections"):
        # Extract section content
        section_text = extract_section_content(text, section['number'], sections)
        
        if not section_text:
            continue
        
        # Extract relationships from this section
        relationships = extractor.extract_relationships(section_text, section['number'])
        
        # Create relationships in graph
        for rel in relationships:
            rel_type = rel['type']
            source = rel['source']
            target = rel['target']
            props = rel['properties']
            
            # Skip if relationship already exists (deduplication)
            if relationship_exists(source, target, rel_type):
                continue
            
            # Map to correct node labels
            if rel_type == 'PROHIBITED_IN':
                # Equipment PROHIBITED_IN Location
                graph.query("""
                MATCH (e:Equipment {name: $source})
                MATCH (l:Location {name: $target})
                MERGE (e)-[r:PROHIBITED_IN {code_ref: $code_ref}]->(l)
                SET r.reason = $reason, r.severity = $severity
                """, {
                    "source": source,
                    "target": target,
                    "code_ref": props['code_ref'],
                    "reason": props.get('reason', ''),
                    "severity": props.get('severity', '')
                })
                relationship_stats[rel_type] += 1
                
            elif rel_type == 'REQUIRES_CLEARANCE':
                # Equipment REQUIRES_CLEARANCE from Material/Location
                # Determine target type
                target_query = """
                OPTIONAL MATCH (m:Material {name: $target})
                OPTIONAL MATCH (l:Location {name: $target})
                WITH COALESCE(m, l) as target_node
                WHERE target_node IS NOT NULL
                MATCH (e:Equipment {name: $source})
                MERGE (e)-[r:REQUIRES_CLEARANCE {code_ref: $code_ref}]->(target_node)
                SET r.min_inches = $min_inches, r.from = $from
                """
                graph.query(target_query, {
                    "source": source,
                    "target": target,
                    "code_ref": props['code_ref'],
                    "min_inches": props.get('min_inches', 0),
                    "from": props.get('from', '')
                })
                relationship_stats[rel_type] += 1
                
            elif rel_type == 'MUST_COMPLY_WITH':
                # Equipment MUST_COMPLY_WITH Standard
                graph.query("""
                MATCH (e:Equipment {name: $source})
                MATCH (st:Standard {name: $target})
                MERGE (e)-[r:MUST_COMPLY_WITH {code_ref: $code_ref}]->(st)
                SET r.test_required = $test_required
                """, {
                    "source": source,
                    "target": target,
                    "code_ref": props['code_ref'],
                    "test_required": props.get('test_required', False)
                })
                relationship_stats[rel_type] += 1
                
            elif rel_type == 'PERMITTED_IN':
                # Equipment PERMITTED_IN Location
                graph.query("""
                MATCH (e:Equipment {name: $source})
                MATCH (l:Location {name: $target})
                MERGE (e)-[r:PERMITTED_IN {code_ref: $code_ref}]->(l)
                SET r.if_condition = $if_condition
                """, {
                    "source": source,
                    "target": target,
                    "code_ref": props['code_ref'],
                    "if_condition": props.get('if_condition', '')
                })
                relationship_stats[rel_type] += 1
                
            elif rel_type == 'REQUIRES_DEVICE':
                # Equipment REQUIRES_DEVICE SafetyDevice
                graph.query("""
                MATCH (e:Equipment {name: $source})
                MATCH (sd:SafetyDevice {name: $target})
                MERGE (e)-[r:REQUIRES_DEVICE {code_ref: $code_ref}]->(sd)
                SET r.mandatory = $mandatory
                """, {
                    "source": source,
                    "target": target,
                    "code_ref": props['code_ref'],
                    "mandatory": props.get('mandatory', False)
                })
                relationship_stats[rel_type] += 1
                
            elif rel_type == 'MADE_OF':
                # Equipment MADE_OF Material
                graph.query("""
                MATCH (e:Equipment {name: $source})
                MATCH (m:Material {name: $target})
                MERGE (e)-[r:MADE_OF {code_ref: $code_ref}]->(m)
                SET r.requirement = $requirement
                """, {
                    "source": source,
                    "target": target,
                    "code_ref": props['code_ref'],
                    "requirement": props.get('requirement', 'optional')
                })
                relationship_stats[rel_type] += 1
    
    print(f"✓ Created domain relationships:")
    for rel_type, count in relationship_stats.items():
        print(f"    {rel_type}: {count}")


def create_text_chunks(text, sections):
    """Add content and embeddings to Section nodes (no separate chunk nodes)"""
    print("\n[8/8] Adding content to Section nodes...")
    
    sections_processed = 0
    for section in tqdm(sections, desc="Section content"):
        # Extract section content
        section_text = extract_section_content(text, section['number'], sections)
        
        if not section_text or len(section_text) < 50:
            continue
        
        # Store content in 'summary' property (primary) and keep 'text' for compatibility
        # Summary is what the agent queries use
        graph.query("""
        MATCH (s:Section {number: $number})
        SET s.summary = $summary,
            s.text = $text
        """, {
            "number": section['number'],
            "summary": section_text,  # Full text in summary
            "text": section_text[:3000]  # Keep first 3000 chars in text for compatibility
        })
        sections_processed += 1
    
    print(f"✓ Added content to {sections_processed} Section nodes")

def create_vector_index():
    """Vector index disabled - using clean graph architecture instead"""
    print("\n[9/9] Skipping vector index (clean graph architecture)...")
    print("✓ Using graph relationships for queries instead of vector search")

def show_statistics():
    """Show database statistics"""
    print("\n" + "="*60)
    print("Database Statistics")
    print("="*60)
    
    stats = {
        "Documents": graph.query("MATCH (d:Document) RETURN count(d) as count")[0]["count"],
        "Chapters": graph.query("MATCH (c:Chapter) RETURN count(c) as count")[0]["count"],
        "Sections": graph.query("MATCH (s:Section) RETURN count(s) as count")[0]["count"],
        "Equipment": graph.query("MATCH (e:Equipment) RETURN count(e) as count")[0]["count"],
        "Locations": graph.query("MATCH (l:Location) RETURN count(l) as count")[0]["count"],
        "Materials": graph.query("MATCH (m:Material) RETURN count(m) as count")[0]["count"],
        "Standards": graph.query("MATCH (s:Standard) RETURN count(s) as count")[0]["count"],
        "SafetyDevices": graph.query("MATCH (sd:SafetyDevice) RETURN count(sd) as count")[0]["count"],
        "Total Nodes": graph.query("MATCH (n) RETURN count(n) as count")[0]["count"],
        "Total Relationships": graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
    }
    
    for label, count in stats.items():
        print(f"  {label:20s}: {count}")
    
    # Show relationship type breakdown
    print("\nRelationship Types:")
    rel_types = graph.query("""
    MATCH ()-[r]->()
    RETURN type(r) as rel_type, count(r) as count
    ORDER BY count DESC
    """)
    for rel in rel_types:
        print(f"  {rel['rel_type']:30s}: {rel['count']}")
    
    print("="*60)

def main():
    """Main execution function"""
    print("="*60)
    print("HVAC GraphRAG Database Population (Phase 2A)")
    print("="*60)
    
    # Display current configuration
    Config.display()
    
    # Extract PDF
    text = extract_pdf_text(Config.PDF_PATH)
    
    # Extract sections
    sections = extract_sections(text)
    
    # Build graph
    create_document_node()
    create_chapter_nodes(sections)
    create_section_nodes(sections)
    create_domain_entity_nodes()  # New: Create Equipment, Location, Material, Standard, SafetyDevice
    create_domain_relationships(text, sections)  # New: Extract relationships using NLP
    
    # Extract and store tables
    print("\n" + "="*60)
    print("TABLE EXTRACTION")
    print("="*60)
    table_extractor = TableExtractor()
    table_extractor.extract_and_store_all_tables(text)
    
    create_text_chunks(text, sections)
    create_vector_index()
    
    # Show stats
    show_statistics()
    
    print("\n✓ Database ready! Run 'streamlit run bot.py' to use chatbot.")
    print("\nTest query: 'What are the standards for installing an Air Handler in a building?'")

if __name__ == "__main__":
    main()
