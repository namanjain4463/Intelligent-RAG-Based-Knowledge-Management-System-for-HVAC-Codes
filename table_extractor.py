"""
Table Extraction Module for HVAC Codes PDF

Extracts tables from the PDF and creates proper graph structure:
- Table nodes (with caption/title)
- TableRow nodes (for each row)
- TableCell nodes (for each cell value)
- Relationships: Section -[:CONTAINS_TABLE]-> Table -[:HAS_ROW]-> TableRow -[:HAS_CELL]-> TableCell
"""
import re
from typing import List, Dict, Tuple
from graph import graph

class TableExtractor:
    """Extract and structure tables from HVAC codes text"""
    
    def __init__(self):
        # Pattern to match table captions (e.g., "TABLE 303.3 PROHIBITED LOCATIONS")
        # We'll use "TABLE_" prefix for table numbers to avoid confusion with Section numbers
        self.table_caption_pattern = re.compile(
            r'TABLE\s+(\d{3}(?:\.\d+)*(?:\(\d+\))?)\s+(.+?)(?:\n|$)',
            re.IGNORECASE | re.MULTILINE
        )
        
    def detect_tables(self, text: str) -> List[Dict]:
        """
        Detect all tables in the text by finding TABLE captions
        
        Returns list of dicts with:
        - table_number: e.g., "303.3", "803.9(2)"
        - caption: table title
        - start_pos: position in text where table starts
        """
        tables = []
        
        for match in self.table_caption_pattern.finditer(text):
            table_number = match.group(1)
            caption = match.group(2).strip()
            start_pos = match.start()
            
            tables.append({
                'table_number': table_number,
                'caption': caption,
                'start_pos': start_pos,
                'end_pos': None  # Will be set later
            })
        
        # Set end positions (from current table to next table or end of section)
        for i in range(len(tables)):
            if i < len(tables) - 1:
                tables[i]['end_pos'] = tables[i + 1]['start_pos']
            else:
                tables[i]['end_pos'] = len(text)
        
        return tables
    
    def extract_table_content(self, text: str, table_info: Dict) -> str:
        """Extract the content of a specific table"""
        start = table_info['start_pos']
        end = table_info['end_pos']
        return text[start:end]
    
    def parse_table_rows(self, table_text: str, table_number: str) -> List[Dict[str, any]]:
        """
        Parse table text into structured rows with better parsing
        
        Returns list of dicts with:
        - row_type: 'header' or 'data'
        - cells: list of cell values
        - raw_text: original text
        """
        lines = table_text.split('\n')
        rows = []
        current_cells = []
        in_header = True
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_cells:
                    rows.append({
                        'row_type': 'header' if in_header else 'data',
                        'cells': current_cells.copy(),
                        'raw_text': ' | '.join(current_cells)
                    })
                    current_cells = []
                    if in_header:
                        in_header = False  # First group is header
                continue
            
            # Skip the table caption line
            if line.upper().startswith('TABLE'):
                continue
            
            # Skip "For SI:" conversion lines
            if line.startswith('For SI:'):
                continue
            
            # Detect header rows (ALL CAPS or title case, no numbers)
            if line.isupper() and not re.search(r'\d+\.?\d*', line):
                if current_cells:
                    rows.append({
                        'row_type': 'header' if in_header else 'data',
                        'cells': current_cells.copy(),
                        'raw_text': ' | '.join(current_cells)
                    })
                    current_cells = []
                current_cells.append(line)
                continue
            
            # Data rows - can contain numbers, measurements, text
            current_cells.append(line)
        
        # Add last row if exists
        if current_cells:
            rows.append({
                'row_type': 'data',
                'cells': current_cells.copy(),
                'raw_text': ' | '.join(current_cells)
            })
        
        return rows
    
    def create_table_nodes(self, section_number: str, tables: List[Dict], full_text: str):
        """
        Create Table and TableRow nodes in Neo4j
        
        Args:
            section_number: The section this table belongs to (e.g., "303.3")
            tables: List of table metadata
            full_text: Full text to extract table content from
        """
        created_count = 0
        
        for table_info in tables:
            table_number = table_info['table_number']
            caption = table_info['caption']
            
            # Create unique table ID to avoid confusion with Section numbers
            # "TABLE_305.4" instead of "305.4" to distinguish from Section 305.4
            table_id = f"TABLE_{table_number}"
            
            # Extract table content
            table_text = self.extract_table_content(full_text, table_info)
            
            # Parse into rows
            rows = self.parse_table_rows(table_text, table_number)
            
            if not rows:
                continue  # Skip empty tables
            
            # Separate headers and data rows
            headers = [r for r in rows if r['row_type'] == 'header']
            data_rows = [r for r in rows if r['row_type'] == 'data']
            
            # Create header text
            header_text = ' | '.join([' '.join(h['cells']) for h in headers]) if headers else ''
            
            # Create Table node with unique ID
            # CRITICAL: Use 'id' as primary key (TABLE_305.4), 'number' for reference (305.4)
            create_table_query = """
            MERGE (t:Table {id: $table_id})
            SET t.number = $table_number,
                t.caption = $caption,
                t.section_number = $section_number,
                t.row_count = $row_count,
                t.data_row_count = $data_row_count,
                t.header = $header,
                t.full_text = $full_text
            """
            
            # Try to link to section if it exists
            link_section_query = """
            MATCH (t:Table {id: $table_id})
            OPTIONAL MATCH (s:Section {number: $section_number})
            FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
                MERGE (s)-[:CONTAINS_TABLE]->(t)
            )
            """
            
            graph.query(create_table_query, {
                'table_id': table_id,
                'table_number': table_number,
                'caption': caption,
                'section_number': section_number,
                'row_count': len(rows),
                'data_row_count': len(data_rows),
                'header': header_text,
                'full_text': table_text[:1000]  # First 1000 chars
            })
            
            graph.query(link_section_query, {
                'table_id': table_id,
                'section_number': section_number
            })
            
            # Create TableRow nodes for data rows only
            for row_idx, row_data in enumerate(data_rows):
                create_row_query = """
                MATCH (t:Table {id: $table_id})
                CREATE (tr:TableRow {
                    table_id: $table_id,
                    table_number: $table_number,
                    row_index: $row_idx,
                    cell_count: $cell_count,
                    cells: $cells,
                    raw_text: $raw_text
                })
                CREATE (t)-[:HAS_ROW]->(tr)
                """
                
                graph.query(create_row_query, {
                    'table_id': table_id,
                    'table_number': table_number,
                    'row_idx': row_idx,
                    'cell_count': len(row_data['cells']),
                    'cells': row_data['cells'],
                    'raw_text': row_data['raw_text']
                })
            
            created_count += 1
        
        return created_count
    
    def extract_and_store_all_tables(self, text: str):
        """
        Main method: Extract all tables from text and store in Neo4j
        
        Args:
            text: Full PDF text content
        """
        print("\n[1/3] Detecting tables in PDF...")
        tables = self.detect_tables(text)
        print(f"✓ Found {len(tables)} tables with TABLE captions")
        
        if not tables:
            print("  ⚠️  No tables found in document")
            return
        
        # Show sample of detected tables
        print(f"\n[2/3] Sample tables detected:")
        for table in tables[:5]:
            print(f"  - TABLE {table['table_number']}: {table['caption'][:60]}...")
        if len(tables) > 5:
            print(f"  ... and {len(tables) - 5} more tables")
        
        # Group tables by section
        tables_by_section = {}
        for table in tables:
            # Extract section number from table number
            # e.g., "303.3" → section "303", "803.9(2)" → section "803.9"
            section_num = table['table_number'].split('(')[0]
            
            if section_num not in tables_by_section:
                tables_by_section[section_num] = []
            tables_by_section[section_num].append(table)
        
        print(f"\n[3/3] Creating Table and TableRow nodes...")
        total_created = 0
        total_rows = 0
        
        from tqdm import tqdm
        for section_num, section_tables in tqdm(tables_by_section.items(), desc="Sections with tables"):
            created = self.create_table_nodes(section_num, section_tables, text)
            total_created += created
            
            # Count rows created
            for table in section_tables:
                table_text = self.extract_table_content(text, table)
                rows = self.parse_table_rows(table_text, table['table_number'])
                data_rows = [r for r in rows if r['row_type'] == 'data']
                total_rows += len(data_rows)
        
        print(f"✓ Created {total_created} Table nodes")
        print(f"✓ Created {total_rows} TableRow nodes")
        print(f"✓ Tables linked to sections with CONTAINS_TABLE relationships")


# Example usage
if __name__ == "__main__":
    extractor = TableExtractor()
    
    # Test with sample table text
    sample_text = """
    TABLE 303.3 PROHIBITED LOCATIONS FOR FURNACES
    Location Type    Allowed    Reason
    Bedrooms         No         Fire hazard
    Bathrooms        No         Moisture damage
    
    TABLE 803.9(2) MINIMUM CHIMNEY CONNECTOR THICKNESS
    Area (sq in)    Gage        Thickness
    0-50            No. 26      0.022
    50-100          No. 24      0.028
    """
    
    tables = extractor.detect_tables(sample_text)
    print(f"Detected {len(tables)} tables:")
    for t in tables:
        print(f"  - Table {t['table_number']}: {t['caption']}")
