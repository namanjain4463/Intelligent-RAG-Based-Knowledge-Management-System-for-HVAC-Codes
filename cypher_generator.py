"""
Cypher Query Generator (Phase 2B - Schema-Aware with Two-Stage Querying)
Uses LLM to generate domain-driven Cypher queries from natural language
"""
import re
from llm import llm
from graph import graph
from langchain.prompts import PromptTemplate
from vocabulary import entity_normalizer

def extract_entities(query: str):
    """
    Extract and normalize entities from user query using vocabulary system.
    Two-stage approach: Entity Recognition → Relationship Query
    
    Args:
        query: Natural language query from user
    
    Returns:
        Dict with normalized equipment, location, material, standard, safety_device, section
    """
    query_lower = query.lower()
    
    # Extract words/phrases from query
    words = query_lower.split()
    phrases = []
    for i in range(len(words)):
        for j in range(i+1, min(i+5, len(words)+1)):
            phrases.append(' '.join(words[i:j]))
    
    # Try to match entities using normalizer
    entities = {
        "equipment": None,
        "location": None,
        "material": None,
        "standard": None,
        "safety_device": None,
        "section": None
    }
    
    # Try each phrase
    for phrase in sorted(phrases, key=len, reverse=True):
        if not entities["equipment"]:
            entities["equipment"] = entity_normalizer.normalize_equipment(phrase, fuzzy_threshold=0.75)
        if not entities["location"]:
            entities["location"] = entity_normalizer.normalize_location(phrase, fuzzy_threshold=0.75)
        if not entities["material"]:
            entities["material"] = entity_normalizer.normalize_material(phrase, fuzzy_threshold=0.75)
        if not entities["standard"]:
            entities["standard"] = entity_normalizer.normalize_standard(phrase, fuzzy_threshold=0.75)
        if not entities["safety_device"]:
            entities["safety_device"] = entity_normalizer.normalize_safety_device(phrase, fuzzy_threshold=0.75)
    
    # Extract section number (e.g., 301, 301.3, 308.4.2)
    section_match = re.search(r'\b(\d{3,4}(?:\.\d+)*)\b', query)
    entities["section"] = section_match.group(1) if section_match else None
    
    return entities

# Cypher Generation Prompt with Phase 2A Domain Schema (30+ Examples)
CYPHER_GENERATION_PROMPT = """You are a Cypher query generator for an HVAC Codes Neo4j knowledge graph.

⚠️ CRITICAL SECURITY RULE: Generate ONLY READ-ONLY queries (MATCH, RETURN, WITH, WHERE, ORDER BY, LIMIT).
NEVER generate CREATE, MERGE, DELETE, SET, REMOVE, or any data modification commands.

DATABASE SCHEMA (Phase 2A - Domain-Driven):

Node Labels:
- Document {{{{name, version, date}}}}
- Chapter {{{{number, title}}}}
- Section {{{{number, path, level, title}}}}
- Equipment {{{{name, type, synonyms[]}}}}  // 34 types: Air Handler, Furnace, Boiler, Water Heater, etc.
- Location {{{{name, type, synonyms[]}}}}  // 27 types: Mechanical Room, Attic, Basement, Bedroom, etc.
- Material {{{{name, type, synonyms[]}}}}  // 21 types: Combustible Material, Steel, Concrete, Wood, etc.
- Standard {{{{name, type, synonyms[]}}}}  // 16 types: UL, NFPA, ASHRAE, ASME, etc.
- SafetyDevice {{{{name, type, synonyms[]}}}}  // 12 types: Relief Valve, Fire Damper, Disconnect Switch, etc.
- Requirement {{{{name, type, code_ref, description, mandatory, applies_to}}}}  // 8 general requirements from Section 301
- TextChunk {{{{id, text, sectionPath, embedding}}}}

Hierarchical Relationships:
- (Document)-[:CONTAINS]->(Chapter)
- (Chapter)-[:CONTAINS]->(Section)
- (Section)-[:CONTAINS]->(Section)  // Nested sections
- (TextChunk)-[:BELONGS_TO]->(Section)

Domain Relationships (WITH SECTION REFERENCES):
- (Equipment)-[:PROHIBITED_IN {{code_ref, reason, severity}}]->(Location)
  // Example: (Furnace)-[:PROHIBITED_IN {{code_ref: "303.3"}}]->(Bedroom)
  
- (Equipment)-[:REQUIRES_CLEARANCE {{code_ref, min_inches, from}}]->(Material|Location)
  // Example: (Air Handler)-[:REQUIRES_CLEARANCE {{code_ref: "308.4", min_inches: 24}}]->(Combustible Material)
  
- (Equipment)-[:MUST_COMPLY_WITH {{code_ref, test_required}}]->(Standard)
  // Example: (Boiler)-[:MUST_COMPLY_WITH {{code_ref: "1004.1"}}]->(ASME)
  
- (Equipment)-[:MUST_MEET {{code_ref}}]->(Requirement)
  // Example: (Boiler)-[:MUST_MEET {{code_ref: "301.7"}}]->(Listed and Labeled)
  // All equipment must meet general Section 301 requirements
  
- (Equipment)-[:PERMITTED_IN {{code_ref, if_condition}}]->(Location)
  // Example: (Gas Appliance)-[:PERMITTED_IN {{code_ref: "303.1", if_condition: "with ventilation"}}]->(Mechanical Room)
  
- (Equipment)-[:REQUIRES_DEVICE {{code_ref, mandatory}}]->(SafetyDevice)
  // Example: (Water Heater)-[:REQUIRES_DEVICE {{code_ref: "1006.3"}}]->(Temperature Relief Valve)

KEY PROPERTIES FOR SECTION TRACEABILITY:
- All domain relationships have 'code_ref' property containing section number
- Use code_ref to link back to Section nodes for full text
- Multiple relationships can reference same section

Hierarchy Info:
- Sections use materialized paths: path="/10/1001/1001.1"
- Sections have levels: level=1,2,3,4
- Use path STARTS WITH for descendant queries

⚠️ CRITICAL RULE FOR SUBSECTION REFERENCES (SECTIONS WITH DECIMAL POINT):
When the user asks about a section number containing a DECIMAL POINT (e.g., "301.5", "303.3", "304.1"),
you MUST query BOTH Section nodes AND Requirement nodes, because:
- Section nodes: Hierarchical sections (303.3, 304.1, 306.3, etc.) created from PDF structure
- Requirement nodes: Specific requirements (301.5, 301.7, 301.9, etc.) with detailed descriptions

⚠️ ALWAYS USE OPTIONAL MATCH FOR BOTH to handle all cases:

CORRECT PATTERN for subsections with dots:
OPTIONAL MATCH (s:Section {{{{number: "303.3"}}}})
OPTIONAL MATCH (r:Requirement {{{{code_ref: "303.3"}}}})
RETURN 
  COALESCE(s.number, r.code_ref) as section_number,
  COALESCE(s.title, r.name) as title,
  COALESCE(s.summary, r.description) as content

EXAMPLES OF CORRECT SUBSECTION QUERIES:
- "What does Section 303.3 say?" → Query BOTH Section(303.3) AND Requirement(303.3)
- "What does Section 301.7 say?" → Query BOTH Section(301.7) AND Requirement(301.7)
- "Tell me about Section 304.1" → Query BOTH Section(304.1) AND Requirement(304.1)

RULE: 
- NO DOT: "301" → MATCH (s:Section {{{{number: "301"}}}}) (whole section only)
- HAS DOT: "301.7" → OPTIONAL MATCH both Section AND Requirement, use COALESCE
- This ensures we NEVER miss data regardless of whether it's stored as Section or Requirement

30+ CYPHER QUERY EXAMPLES (DOMAIN-DRIVEN):

Example 1: Get specific section by number (NO DOT - whole section)
Query: "What is Section 901?"
Cypher:
MATCH (s:Section {{{{{{{{number: "901"}}}}}}}})
RETURN s.number, s.title, s.summary, s.level, s.path

Example 1a: Get specific SUBsection by number (HAS DOT - query both Section and Requirement)
Query: "What is Section 303.3?"
Cypher:
OPTIONAL MATCH (s:Section {{{{{{{{number: "303.3"}}}}}}}})
OPTIONAL MATCH (r:Requirement {{{{{{{{code_ref: "303.3"}}}}}}}})
RETURN 
  COALESCE(s.number, r.code_ref) as section_number,
  COALESCE(s.title, r.name) as title,
  COALESCE(s.summary, r.description) as content,
  CASE WHEN s IS NOT NULL THEN s.level ELSE null END as level

Example 1b: Get section WITH CONTEXT (when section has no summary, show relationships referencing it)
Query: "What is Section 303.3 about?"
Cypher:
// Get the section info
OPTIONAL MATCH (s:Section {{{{{{{{number: "303.3"}}}}}}}})
// Get equipment prohibited per this section
OPTIONAL MATCH (e:Equipment)-[r:PROHIBITED_IN]->(l:Location)
WHERE r.code_ref = "303.3"
// Get equipment permitted per this section
OPTIONAL MATCH (e2:Equipment)-[r2:PERMITTED_IN]->(l2:Location)
WHERE r2.code_ref = "303.3"
// Get subsections under this section
OPTIONAL MATCH (s)-[:CONTAINS]->(child:Section)
RETURN 
  s.number as section_number,
  s.title as title,
  collect(DISTINCT {{{{{{{{equipment: e.name, location: l.name, reason: r.reason, type: 'PROHIBITED'}}}}}}}}) as prohibitions,
  collect(DISTINCT {{{{{{{{equipment: e2.name, location: l2.name, condition: r2.if_condition, type: 'PERMITTED'}}}}}}}}) as permissions,
  collect(DISTINCT child.number) as subsections

Example 2: Get all subsections (descendants)
Query: "What subsections are in Section 901?"
Cypher:
MATCH (parent:Section {{{{{{{{number: "901"}}}}}}}})-[:CONTAINS*]->(child:Section)
RETURN child.number, child.title, child.level
ORDER BY child.path

Example 2a: Get all subsections under a Chapter
Query: "What are all the subsections under Chapter 3?"
Cypher:
MATCH (c:Chapter {{{{{{{{number: "3"}}}}}}}})-[:CONTAINS*]->(s:Section)
RETURN s.number, s.title, s.level
ORDER BY s.number

Example 2b: Get direct children of a Chapter (Section level 1 only)
Query: "What are the main sections in Chapter 3?"
Cypher:
MATCH (c:Chapter {{{{{{{{number: "3"}}}}}}}})-[:CONTAINS]->(s:Section)
WHERE s.level = 1
RETURN s.number, s.title
ORDER BY s.number

Example 3: Get direct children only
Query: "What are the direct subsections of 901?"
Cypher:
MATCH (parent:Section {{{{{{{{number: "901"}}}}}}}})-[:CONTAINS]->(child:Section)
RETURN child.number, child.title
ORDER BY child.number

Example 4: Find where equipment is PROHIBITED
Query: "Where can't I install a Furnace?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Furnace"}}}}}}}})-[r:PROHIBITED_IN]->(l:Location)
RETURN l.name as location, r.code_ref as section, r.reason as reason
ORDER BY r.code_ref

Example 5: Find clearance requirements for equipment
Query: "What clearances does an Air Handler need?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Air Handler"}}}}}}}})-[r:REQUIRES_CLEARANCE]->(target)
RETURN labels(target)[0] as target_type, target.name as from_what, 
       r.min_inches as inches, r.code_ref as section
ORDER BY r.code_ref

Example 6: Find compliance standards for equipment (SPECIFIC standards only)
Query: "What specific standards does a Boiler need to comply with?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Boiler"}}}}}}}})-[r:MUST_COMPLY_WITH]->(s:Standard)
RETURN s.name as standard, r.code_ref as section, r.test_required as test_required
ORDER BY r.code_ref

Example 6a: Find general requirements for equipment
Query: "What are the general requirements for a Boiler?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Boiler"}}}}}}}})-[r:MUST_MEET]->(req:Requirement)
RETURN req.name as requirement, req.type as category, req.code_ref as section, req.description
ORDER BY req.type, req.name

Example 6b: Find ALL standards and requirements (COMPREHENSIVE)
Query: "What standards must a Boiler comply with?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Boiler"}}}}}}}})
OPTIONAL MATCH (e)-[r1:MUST_COMPLY_WITH]->(s:Standard)
OPTIONAL MATCH (e)-[r2:MUST_MEET]->(req:Requirement)
RETURN 
  collect(DISTINCT {{{{{{{{standard: s.name, section: r1.code_ref, test_required: r1.test_required}}}}}}}}) as specific_standards,
  collect(DISTINCT {{{{{{{{requirement: req.name, type: req.type, section: req.code_ref, description: req.description}}}}}}}}) as general_requirements

Example 7: Find safety devices required for equipment
Query: "What safety devices does a Water Heater require?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Water Heater"}}}}}}}})-[r:REQUIRES_DEVICE]->(sd:SafetyDevice)
RETURN sd.name as device, r.code_ref as section, r.mandatory as mandatory
ORDER BY r.code_ref

Example 8: Find permitted locations for equipment
Query: "Where can I install a Gas Appliance?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Gas Appliance"}}}}}}}})-[r:PERMITTED_IN]->(l:Location)
RETURN l.name as location, r.if_condition as condition, r.code_ref as section
ORDER BY r.code_ref

Example 8a: Find permitted locations for equipment (alternative phrasing)
Query: "Where can an Air Handler be installed?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Air Handler"}}}}}}}})-[r:PERMITTED_IN]->(l:Location)
RETURN l.name as location, r.if_condition as condition, r.code_ref as section
ORDER BY r.code_ref

Example 8b: Find permitted locations with BOTH permitted and prohibited
Query: "What locations are allowed for installing a Furnace?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Furnace"}}}}}}}})
OPTIONAL MATCH (e)-[r1:PERMITTED_IN]->(loc1:Location)
OPTIONAL MATCH (e)-[r2:PROHIBITED_IN]->(loc2:Location)
RETURN 
  collect(DISTINCT {{{{{{{{location: loc1.name, section: r1.code_ref, status: 'PERMITTED'}}}}}}}}) as permitted_locations,
  collect(DISTINCT {{{{{{{{location: loc2.name, section: r2.code_ref, status: 'PROHIBITED'}}}}}}}}) as prohibited_locations

Example 9: Multi-dimensional equipment query (ALL requirements)
Query: "What are all the requirements for installing an Air Handler?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Air Handler"}}}}}}}})
OPTIONAL MATCH (e)-[r1:PROHIBITED_IN]->(loc1:Location)
OPTIONAL MATCH (e)-[r2:REQUIRES_CLEARANCE]->(target)
OPTIONAL MATCH (e)-[r3:MUST_COMPLY_WITH]->(std:Standard)
OPTIONAL MATCH (e)-[r4:REQUIRES_DEVICE]->(dev:SafetyDevice)
OPTIONAL MATCH (e)-[r5:PERMITTED_IN]->(loc2:Location)
OPTIONAL MATCH (e)-[r6:MUST_MEET]->(req:Requirement)
RETURN 
  collect(DISTINCT {{{{{{{{type: 'PROHIBITED_IN', location: loc1.name, section: r1.code_ref}}}}}}}}) as prohibitions,
  collect(DISTINCT {{{{{{{{type: 'CLEARANCE', from: target.name, inches: r2.min_inches, section: r2.code_ref}}}}}}}}) as clearances,
  collect(DISTINCT {{{{{{{{type: 'STANDARD', standard: std.name, section: r3.code_ref}}}}}}}}) as standards,
  collect(DISTINCT {{{{{{{{type: 'DEVICE', device: dev.name, section: r4.code_ref}}}}}}}}) as devices,
  collect(DISTINCT {{{{{{{{type: 'PERMITTED', location: loc2.name, condition: r5.if_condition, section: r5.code_ref}}}}}}}}) as permissions,
  collect(DISTINCT {{{{{{{{type: 'REQUIREMENT', requirement: req.name, category: req.type, section: req.code_ref}}}}}}}}) as general_requirements

Example 10: Get section details and summary
Query: "What does Section 303 cover?"
Cypher:
MATCH (s:Section {{{{{{{{number: "303"}}}}}}}})
RETURN s.number, s.title, s.summary, s.text, s.level, s.path

Example 10a: Get section details from code_ref
Query: "Show full text for Section 303.3"
Cypher:
MATCH (s:Section {{{{{{{{number: "303.3"}}}}}}}})
RETURN s.number, s.title, s.text, s.summary, s.path

Example 10b: Find section by title keyword
Query: "What section covers equipment location?"
Cypher:
MATCH (s:Section)
WHERE toLower(s.title) CONTAINS 'equipment' AND toLower(s.title) CONTAINS 'location'
RETURN s.number, s.title, s.summary
ORDER BY s.number

Example 10c: Get requirement details by section reference (for subsections like 301.7)
Query: "What does Section 301.7 say about listing and labeling?"
Cypher:
MATCH (r:Requirement {{{{{{{{code_ref: "301.7"}}}}}}}})
RETURN r.name as requirement, r.description as details, r.code_ref as section, r.type as category

Example 10d: Find requirements by section code reference
Query: "What are the requirements in Section 301.12?"
Cypher:
MATCH (r:Requirement)
WHERE r.code_ref = "301.12"
RETURN r.name as requirement, r.description as details, r.code_ref as section, r.type as category
ORDER BY r.name

Example 10e: Get requirements and find which equipment must comply
Query: "What does Section 301.7 say and which equipment must comply?"
Cypher:
MATCH (r:Requirement {{{{{{{{code_ref: "301.7"}}}}}}}})
OPTIONAL MATCH (e:Equipment)-[:MUST_MEET]->(r)
RETURN r.name as requirement, r.description as details, r.code_ref as section, 
       collect(DISTINCT e.name) as equipment_list

Example 10f: Hybrid search for section content (checks both Section nodes and Requirements)
Query: "What does Section 301.7 say about listing and labeling, and which equipment must comply?"
Cypher:
// First try to find as Section node
OPTIONAL MATCH (s:Section {{{{{{{{number: "301.7"}}}}}}}})
// Also try to find as Requirement by code_ref
OPTIONAL MATCH (r:Requirement {{{{{{{{code_ref: "301.7"}}}}}}}})
OPTIONAL MATCH (e:Equipment)-[:MUST_MEET]->(r)
RETURN 
  CASE WHEN s IS NOT NULL THEN s.number ELSE r.code_ref END as section_number,
  CASE WHEN s IS NOT NULL THEN s.title ELSE r.name END as title,
  CASE WHEN s IS NOT NULL THEN s.summary ELSE r.description END as content,
  collect(DISTINCT e.name) as equipment_that_must_comply

Example 11: Find all equipment prohibited in a location
Query: "What equipment is prohibited in Bedrooms?"
Cypher:
MATCH (e:Equipment)-[r:PROHIBITED_IN]->(l:Location {{{{{{{{name: "Bedroom"}}}}}}}})
RETURN e.name as equipment, r.code_ref as section, r.reason as reason
ORDER BY e.name

Example 12: Find requirements by keyword search
Query: "What altitude considerations exist?"
Cypher:
MATCH (r:Requirement)
WHERE toLower(r.name) CONTAINS 'altitude' OR toLower(r.description) CONTAINS 'altitude'
RETURN r.name, r.description, r.code_ref as section, r.type as category
ORDER BY r.code_ref

Example 12a: Find requirements by type
Query: "What design requirements must equipment meet?"
Cypher:
MATCH (r:Requirement {{{{{{{{type: "Design"}}}}}}}})
RETURN r.name, r.description, r.code_ref as section
ORDER BY r.code_ref

Example 12b: Find equipment that must meet a specific requirement
Query: "What equipment must be listed and labeled?"
Cypher:
MATCH (e:Equipment)-[rel:MUST_MEET]->(r:Requirement)
WHERE toLower(r.name) CONTAINS 'list' OR toLower(r.name) CONTAINS 'label'
RETURN e.name as equipment, r.name as requirement, r.code_ref as section
ORDER BY e.name

Example 12: Find equipment requiring specific clearance
Query: "What equipment needs clearance from combustible materials?"
Cypher:
MATCH (e:Equipment)-[r:REQUIRES_CLEARANCE]->(m:Material {{{{{{{{name: "Combustible Material"}}}}}}}})
RETURN e.name as equipment, r.min_inches as inches, r.code_ref as section
ORDER BY r.min_inches DESC

Example 13: Find materials that can be used for a component
Query: "What materials can be used for ductwork?"
Cypher:
MATCH (e)-[r:MADE_OF]->(m:Material)
WHERE toLower(e.name) CONTAINS 'duct'
RETURN DISTINCT m.name as material, r.code_ref as section
ORDER BY m.name

Example 13a: Find materials for equipment or components (alternative)
Query: "What materials are allowed for piping?"
Cypher:
MATCH (e)-[r:MADE_OF]->(m:Material)
WHERE toLower(e.name) CONTAINS 'pipe' OR toLower(e.name) CONTAINS 'piping'
RETURN DISTINCT m.name as material, r.code_ref as section
ORDER BY m.name

Example 14: List all equipment types
Query: "What equipment is in the database?"
Cypher:
MATCH (e:Equipment)
RETURN e.name, e.type, e.synonyms
ORDER BY e.name

Example 15: List all locations
Query: "What locations are defined?"
Cypher:
MATCH (l:Location)
RETURN l.name, l.type, l.synonyms
ORDER BY l.name

Example 15: List all safety devices
Query: "What safety devices are required?"
Cypher:
MATCH (sd:SafetyDevice)
RETURN sd.name, sd.type
ORDER BY sd.name

Example 16: Find equipment by standard compliance
Query: "What equipment must comply with ASME standards?"
Cypher:
MATCH (e:Equipment)-[r:MUST_COMPLY_WITH]->(s:Standard {{{{{{{{name: "ASME"}}}}}}}})
RETURN e.name, r.code_ref as section
ORDER BY e.name

Example 17: Get all chapters
Query: "List all chapters"
Cypher:
MATCH (c:Chapter)
RETURN c.number, c.title
ORDER BY c.number

Example 18: Get sections at specific depth level
Query: "Show level 3 sections"
Cypher:
MATCH (s:Section)
WHERE s.level = 3
RETURN s.number, s.title
ORDER BY s.number

Example 19: Find sections using path prefix
Query: "All sections under Chapter 3"
Cypher:
MATCH (s:Section)
WHERE s.path STARTS WITH "/3/"
RETURN s.number, s.title
ORDER BY s.path

Example 20: Get ancestors of a section
Query: "What are the parent sections of 303.3?"
Cypher:
MATCH (s:Section {{{{{{{{number: "303.3"}}}}}}}})
UNWIND split(s.path, '/') AS ancestor_num
WITH ancestor_num WHERE ancestor_num <> ''
MATCH (a:Section {{{{{{{{number: ancestor_num}}}}}}}})
RETURN a.number, a.title, a.level
ORDER BY a.level

⚠️ CRITICAL FOR STANDARDS/COMPLIANCE QUESTIONS:
When the user asks "What standards must [equipment] comply with?" or similar:
- Query BOTH MUST_COMPLY_WITH (specific standards like ASME, UL 726) 
- AND MUST_MEET (general requirements like Listed and Labeled, Nameplate, Testing)
- Use the pattern from Example 6b with OPTIONAL MATCH for both relationship types
- This ensures comprehensive answers with BOTH specific standards AND general Section 301 requirements

⚠️ CRITICAL FOR AGGREGATE QUERIES (ACROSS ALL / EVERY / ALL EQUIPMENT):
When user asks "across ALL equipment", "for ALL X", "what do ALL", "common to EVERY":

Example A1: Find safety devices required by ALL equipment
Query: "What safety devices are required across all equipment?"
Cypher:
MATCH (e:Equipment)-[r:REQUIRES_DEVICE]->(sd:SafetyDevice)
WITH sd, collect(DISTINCT e.name) as equipment_list, count(DISTINCT e) as equipment_count
MATCH (total:Equipment)
WITH sd, equipment_list, equipment_count, count(DISTINCT total) as total_equipment
RETURN sd.name as device, 
       equipment_count as required_by_count,
       total_equipment as total_equipment_count,
       CASE WHEN equipment_count = total_equipment 
            THEN 'UNIVERSAL' 
            ELSE 'SELECTIVE (' + toString(equipment_count) + '/' + toString(total_equipment) + ')' 
       END as coverage,
       equipment_list
ORDER BY equipment_count DESC

Example A2: Find locations where ALL equipment is prohibited
Query: "Which locations prohibit all HVAC equipment?"
Cypher:
MATCH (l:Location)<-[r:PROHIBITED_IN]-(e:Equipment)
WITH l, collect(DISTINCT e.name) as prohibited_equipment, count(DISTINCT e) as prohibited_count
MATCH (total:Equipment)
WITH l, prohibited_equipment, prohibited_count, count(DISTINCT total) as total_equipment
WHERE prohibited_count = total_equipment  // Only locations that prohibit EVERY equipment
RETURN l.name as location, 
       prohibited_count + '/' + total_equipment as coverage,
       prohibited_equipment
ORDER BY l.name

Example A3: Find materials required by ALL equipment
Query: "What materials are commonly required across all equipment?"
Cypher:
MATCH (e:Equipment)-[r:MADE_OF]->(m:Material)
WITH m, count(DISTINCT e) as equipment_count
MATCH (total:Equipment)
WITH m, equipment_count, count(DISTINCT total) as total_equipment
WHERE equipment_count >= total_equipment * 0.5  // Materials used by at least 50% of equipment
RETURN m.name as material, 
       equipment_count as used_by_count,
       total_equipment as total_equipment,
       toString(round(equipment_count * 100.0 / total_equipment)) + '%' as percentage
ORDER BY equipment_count DESC

Example A4: Find clearances required across equipment types
Query: "What clearance requirements apply to all equipment?"
Cypher:
MATCH (e:Equipment)-[r:REQUIRES_CLEARANCE]->(target)
WITH type(r) as rel_type, labels(target)[0] as target_type, target.name as target_name, 
     avg(r.min_inches) as avg_inches, collect(DISTINCT e.name) as equipment_list,
     count(DISTINCT e) as equipment_count
RETURN target_type + ': ' + target_name as clearance_from,
       round(avg_inches) as average_inches,
       equipment_count as applies_to_count,
       equipment_list
ORDER BY equipment_count DESC, avg_inches DESC

⚠️ CRITICAL FOR SPECIFIC SUBSECTION REFERENCES IN ANSWERS:
When returning section references (r.code_ref), ALWAYS include the Section title for context:
- Query pattern: MATCH (s:Section {{{{{{{{number: r.code_ref}}}}}}}}) to fetch the title
- Return format: "per Section 303.3 (Prohibited Locations)" NOT just "per Section 303.3"
- This provides specific, nested subsection context (e.g., 303.3 instead of generic 303)

Example with subsection titles:
Query: "Why can't I install a furnace in a bedroom?"
Cypher:
MATCH (e:Equipment {{{{{{{{name: "Furnace"}}}}}}}})-[r:PROHIBITED_IN]->(l:Location {{{{{{{{name: "Bedroom"}}}}}}}})
MATCH (s:Section {{{{{{{{number: r.code_ref}}}}}}}})
RETURN e.name + ' is prohibited in ' + l.name as answer,
       r.reason as reason,
       'per Section ' + s.number + ' (' + s.title + ')' as section_reference

⚠️ TABLE QUERIES - NEW SCHEMA ADDITIONS:

Node Labels for Tables:
- Table {{{{{{{{id, number, caption, section_number, row_count, data_row_count, header, full_text}}}}}}}}
  // CRITICAL: 'id' is primary key (e.g., "TABLE_305.4") to avoid confusion with Section numbers
  // 'number' is reference number (e.g., "305.4") for display
- TableRow {{{{{{{{table_id, table_number, row_index, cell_count, cells[], raw_text}}}}}}}}

Table Relationships:
- (Section)-[:CONTAINS_TABLE]->(Table)  // Section contains tables
- (Table)-[:HAS_ROW]->(TableRow)  // Table has rows

⚠️ CRITICAL: Tables use 'id' as unique identifier (TABLE_305.4) to distinguish from Section 305.4

Example T1: Find tables in a section
Query: "Show me tables in Section 303.3"
Cypher:
MATCH (s:Section {{{{{{{{number: "303.3"}}}}}}}})-[:CONTAINS_TABLE]->(t:Table)
RETURN t.id as table_id, t.number as table_ref, t.caption as title, t.row_count as rows

Example T2: Get table content with rows
Query: "What does TABLE 803.9(2) contain?"
Cypher:
MATCH (t:Table {{{{{{{{id: "TABLE_803.9(2)"}}}}}}}})-[:HAS_ROW]->(tr:TableRow)
RETURN t.caption as title, t.header as columns, 
       collect(tr.raw_text) as data_rows
ORDER BY tr.row_index

Example T3: Find tables by caption keyword
Query: "Show me tables about gauge thickness"
Cypher:
MATCH (t:Table)
WHERE toLower(t.caption) CONTAINS 'gauge' OR toLower(t.caption) CONTAINS 'thickness'
RETURN t.id, t.number, t.caption, t.section_number
ORDER BY t.number

Example T4: Get table with section context
Query: "What are the minimum connector thickness requirements?"
Cypher:
MATCH (s:Section)-[:CONTAINS_TABLE]->(t:Table)
WHERE toLower(t.caption) CONTAINS 'connector' AND toLower(t.caption) CONTAINS 'thickness'
MATCH (t)-[:HAS_ROW]->(tr:TableRow)
RETURN s.number as section, t.number as table_ref, t.caption as table_title,
       collect(tr.cells) as table_data
ORDER BY tr.row_index

Example T5: Search table cell values
Query: "What gauge is required for 10-inch ducts?"
Cypher:
MATCH (t:Table)-[:HAS_ROW]->(tr:TableRow)
WHERE any(cell IN tr.cells WHERE toLower(cell) CONTAINS '10') 
  AND toLower(t.caption) CONTAINS 'duct'
RETURN t.id, t.number, t.caption, tr.cells as matching_row, tr.row_index

Example T6: Distinguish between Section 305.4 and TABLE 305.4
Query: "What is the difference between Section 305.4 and TABLE 305.4?"
Cypher:
OPTIONAL MATCH (s:Section {{{{{{{{number: "305.4"}}}}}}}})
OPTIONAL MATCH (t:Table {{{{{{{{id: "TABLE_305.4"}}}}}}}})
RETURN 
  s.number as section_number,
  s.title as section_title,
  s.summary as section_content,
  t.number as table_reference,
  t.caption as table_title,
  t.row_count as table_rows

USER QUERY: {query}

EXTRACTED ENTITIES (from vocabulary system):
- Equipment: {equipment}
- Location: {location}
- Material: {material}
- Standard: {standard}
- SafetyDevice: {safety_device}
- Section: {section}

TWO-STAGE QUERYING APPROACH:
1. Entity Recognition: Use normalized entity names from vocabulary (e.g., "air handler" → "Air Handler")
2. Relationship Traversal: Query domain relationships (PROHIBITED_IN, REQUIRES_CLEARANCE, etc.) with section references

⚠️ CRITICAL: When querying relationship properties (-[r:...]-), ALWAYS include r.code_ref in the RETURN clause!

CORRECT EXAMPLES:
✅ MATCH (e:Equipment {{name: "Furnace"}})-[r:PROHIBITED_IN]->(l:Location) RETURN l.name, r.code_ref
✅ MATCH (e:Equipment {{name: "Boiler"}})-[r:REQUIRES_CLEARANCE]->(m) RETURN m.name, r.min_inches, r.code_ref
✅ MATCH (e:Equipment)-[r:REQUIRES_DEVICE]->(sd:SafetyDevice) RETURN sd.name, r.code_ref

INCORRECT EXAMPLES (missing code_ref):
❌ MATCH (e:Equipment {{name: "Furnace"}})-[r:PROHIBITED_IN]->(l:Location) RETURN l.name
❌ MATCH (e:Equipment)-[r:MUST_COMPLY_WITH]->(s:Standard) RETURN s.name

Generate ONLY the Cypher query (no explanation, no markdown formatting):
"""

# Create prompt template
cypher_template = PromptTemplate(
    template=CYPHER_GENERATION_PROMPT,
    input_variables=["query", "equipment", "location", "material", "standard", "safety_device", "section"]
)

def generate_cypher(query: str) -> str:
    """
    Generate Cypher query from natural language using LLM (Phase 2B - Schema-Aware)
    Two-stage approach: Entity Recognition → Domain Relationship Traversal
    
    Args:
        query: Natural language query
    
    Returns:
        Generated Cypher query string with section references
    """
    import re
    
    # PRE-PROCESSING: Detect subsection patterns (e.g., "Section 301.7", "304.1")
    # If found, transform query to explicitly ask about Requirements
    subsection_pattern = r'\b(\d+\.\d+)\b'
    match = re.search(subsection_pattern, query)
    
    if match:
        subsection_num = match.group(1)
        # Transform query to explicitly mention "requirement" instead of "section"
        # This helps the LLM understand to query Requirement nodes
        query_hints = query.lower()
        if any(word in query_hints for word in ['what does section', 'tell me about section', 'what is section']):
            # Append explicit hint about querying requirements
            query = query + f" (Note: Check Requirement nodes with code_ref '{subsection_num}')"
    
    # Stage 1: Extract and normalize entities using vocabulary
    entities = extract_entities(query)
    
    # Format prompt with extracted entities
    prompt = cypher_template.format(
        query=query,
        equipment=entities["equipment"] or "None",
        location=entities["location"] or "None",
        material=entities["material"] or "None",
        standard=entities["standard"] or "None",
        safety_device=entities["safety_device"] or "None",
        section=entities["section"] or "None"
    )
    
    # Stage 2: Generate Cypher query using LLM with schema context
    response = llm.invoke(prompt)  # Updated from deprecated predict() to invoke()
    
    # Extract content from response
    cypher = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    
    # Clean up response (remove markdown if present)
    if cypher.startswith("```"):
        # Extract from code block
        lines = cypher.split("\n")
        cypher = "\n".join([l for l in lines if not l.startswith("```")])
        cypher = cypher.replace("cypher", "").strip()
    
    return cypher

def execute_cypher(cypher: str):
    """
    Execute Cypher query against Neo4j (READ-ONLY)
    
    Args:
        cypher: Cypher query string
    
    Returns:
        Dict with success status, results, and cypher
    """
    try:
        # SECURITY: Validate that query is read-only
        if not is_read_only_query(cypher):
            return {
                "success": False,
                "error": "Query rejected: Only READ-ONLY queries are allowed. Query contains forbidden keywords: CREATE, MERGE, DELETE, SET, REMOVE, or DETACH.",
                "cypher": cypher
            }
        
        results = graph.query(cypher)
        return {
            "success": True,
            "results": results,
            "cypher": cypher
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "cypher": cypher
        }

def is_read_only_query(cypher: str) -> bool:
    """
    Validate that a Cypher query is read-only (no modifications)
    
    Args:
        cypher: Cypher query string
    
    Returns:
        True if read-only, False if contains modification keywords
    """
    # Convert to uppercase for case-insensitive matching
    cypher_upper = cypher.upper()
    
    # List of forbidden keywords that modify data
    forbidden_keywords = [
        'CREATE',
        'MERGE', 
        'DELETE',
        'SET',
        'REMOVE',
        'DETACH',
        'DROP',
        'CALL DB.INDEX.FULLTEXT.CREATEINDEX',
        'CALL DB.INDEX.VECTOR.CREATEINDEX',
        'ALTER',
        'RENAME'
    ]
    
    # Check for forbidden keywords
    # Use word boundaries to avoid false positives (e.g., "CREATED_AT" field name)
    for keyword in forbidden_keywords:
        # Check if keyword appears as a standalone word
        if re.search(r'\b' + keyword + r'\b', cypher_upper):
            return False
    
    return True

def query_with_cypher(natural_language_query: str):
    """
    Full pipeline: Generate and execute Cypher from natural language
    
    Args:
        natural_language_query: User's natural language question
    
    Returns:
        Dict with cypher, results, and metadata
    """
    # Generate Cypher
    cypher = generate_cypher(natural_language_query)
    
    # Execute
    result = execute_cypher(cypher)
    
    # Add original query to result
    result["original_query"] = natural_language_query
    result["entities"] = extract_entities(natural_language_query)
    
    return result
