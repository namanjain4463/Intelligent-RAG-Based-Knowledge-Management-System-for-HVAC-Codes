"""
Relationship Extractor using spaCy NLP for HVAC code domain.
Extracts domain-specific relationships from regulatory text.
"""

import re
from typing import List, Dict, Tuple, Optional
import spacy
from vocabulary import entity_normalizer


class RelationshipExtractor:
    """
    Extracts domain relationships from HVAC regulatory text using NLP and pattern matching.
    Combines regex patterns with spaCy dependency parsing for higher accuracy.
    """
    
    # Regulatory pattern keywords
    PROHIBITION_PATTERNS = [
        r'\bshall not\b',
        r'\bshall be prohibited\b',
        r'\bprohibited\b',
        r'\bnot permitted\b',
        r'\bnot allowed\b',
        r'\bmust not\b',
    ]
    
    REQUIREMENT_PATTERNS = [
        r'\bshall\b',
        r'\bmust\b',
        r'\brequired\b',
        r'\bnecessary\b',
        r'\bmandatory\b',
    ]
    
    PERMISSION_PATTERNS = [
        r'\bmay\b',
        r'\bpermitted\b',
        r'\ballowed\b',
        r'\bshall be permitted\b',
    ]
    
    CLEARANCE_PATTERNS = [
        r'clearance.*?(\d+)\s*(?:inch|inches|feet|ft)',
        r'(\d+)\s*(?:inch|inches|feet|ft).*?clearance',
        r'minimum.*?(\d+)\s*(?:inch|inches|feet|ft)',
        r'not less than\s*(\d+)\s*(?:inch|inches|feet|ft)',
    ]
    
    COMPLIANCE_PATTERNS = [
        r'comply with\s+([A-Z]{2,}[\s\d\-\.]*)',
        r'in accordance with\s+([A-Z]{2,}[\s\d\-\.]*)',
        r'conforming to\s+([A-Z]{2,}[\s\d\-\.]*)',
        r'listed and labeled\s+(?:in accordance with|to)\s+([A-Z]{2,}[\s\d\-\.]*)',
        r'shall be listed.*?([A-Z]{2,4})',  # Broader pattern for "shall be listed UL"
        r'complying with\s+([A-Z]{2,}[\s\d\-\.]*)',
        r'meet.*?(?:requirements of|standards of)\s+([A-Z]{2,}[\s\d\-\.]*)',
    ]
    
    MATERIAL_USAGE_PATTERNS = [
        r'(\w+(?:\s+\w+)?)\s+(?:shall be|must be|to be)\s+(?:made of|constructed of|fabricated from)\s+(\w+)',
        r'(\w+)\s+(?:pipe|piping|duct|ductwork|tubing)\s+(?:shall|must)\s+be\s+(\w+)',
        r'(\w+)\s+(?:pipe|piping|duct|ductwork)\s*[,:]?\s*(\w+(?:\s+\w+)?)\s+(?:pipe|piping|duct)',
        r'(\w+)\s+(?:constructed|fabricated|made)\s+(?:of|from)\s+(\w+)',
    ]
    
    def __init__(self):
        """Initialize the relationship extractor with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        self.normalizer = entity_normalizer
    
    def extract_relationships(self, text: str, section_number: str) -> List[Dict]:
        """
        Extract all domain relationships from text.
        
        Args:
            text: Regulatory text to analyze
            section_number: Section reference (e.g., "301.3")
            
        Returns:
            List of relationship dictionaries with type, source, target, properties
        """
        relationships = []
        
        # Split into sentences for better analysis
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        for sentence in sentences:
            # Extract prohibitions
            relationships.extend(self._extract_prohibitions(sentence, section_number))
            
            # Extract clearance requirements
            relationships.extend(self._extract_clearances(sentence, section_number))
            
            # Extract compliance requirements
            relationships.extend(self._extract_compliance(sentence, section_number))
            
            # Extract permissions
            relationships.extend(self._extract_permissions(sentence, section_number))
            
            # Extract device requirements
            relationships.extend(self._extract_device_requirements(sentence, section_number))
            
            # Extract material usage
            relationships.extend(self._extract_material_usage(sentence, section_number))
        
        return relationships
    
    def _extract_prohibitions(self, sentence: str, section_number: str) -> List[Dict]:
        """Extract PROHIBITED_IN relationships."""
        relationships = []
        
        # Check if sentence contains prohibition language
        is_prohibition = any(re.search(pattern, sentence, re.IGNORECASE) 
                           for pattern in self.PROHIBITION_PATTERNS)
        
        if not is_prohibition:
            return relationships
        
        # Extract equipment and locations
        equipment_terms = self._find_equipment_in_text(sentence)
        location_terms = self._find_locations_in_text(sentence)
        
        for equipment in equipment_terms:
            for location in location_terms:
                relationships.append({
                    'type': 'PROHIBITED_IN',
                    'source': equipment,
                    'target': location,
                    'properties': {
                        'code_ref': section_number,
                        'reason': self._extract_reason(sentence),
                        'severity': 'mandatory'
                    }
                })
        
        return relationships
    
    def _extract_clearances(self, sentence: str, section_number: str) -> List[Dict]:
        """Extract REQUIRES_CLEARANCE relationships."""
        relationships = []
        
        # Look for clearance measurements
        clearance_match = None
        for pattern in self.CLEARANCE_PATTERNS:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                clearance_match = match
                break
        
        if not clearance_match:
            return relationships
        
        # Extract distance
        distance_str = clearance_match.group(1)
        distance = int(distance_str)
        
        # Determine unit
        unit = 'inches'
        if 'feet' in sentence.lower() or 'ft' in sentence.lower():
            unit = 'feet'
            distance = distance * 12  # Convert to inches
        
        # Extract equipment and materials/locations
        equipment_terms = self._find_equipment_in_text(sentence)
        material_terms = self._find_materials_in_text(sentence)
        location_terms = self._find_locations_in_text(sentence)
        
        targets = material_terms + location_terms
        
        for equipment in equipment_terms:
            for target in targets:
                relationships.append({
                    'type': 'REQUIRES_CLEARANCE',
                    'source': equipment,
                    'target': target,
                    'properties': {
                        'min_inches': distance,
                        'code_ref': section_number,
                        'from': target
                    }
                })
        
        return relationships
    
    def _extract_compliance(self, sentence: str, section_number: str) -> List[Dict]:
        """Extract MUST_COMPLY_WITH relationships."""
        relationships = []
        
        # Look for compliance references
        standard_match = None
        for pattern in self.COMPLIANCE_PATTERNS:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                standard_match = match
                break
        
        if not standard_match:
            return relationships
        
        # Extract standard reference
        standard_ref = standard_match.group(1).strip()
        
        # Normalize standard
        canonical_standard = self.normalizer.normalize_standard(standard_ref)
        if not canonical_standard:
            canonical_standard = standard_ref  # Use as-is if not recognized
        
        # Extract equipment
        equipment_terms = self._find_equipment_in_text(sentence)
        
        for equipment in equipment_terms:
            relationships.append({
                'type': 'MUST_COMPLY_WITH',
                'source': equipment,
                'target': canonical_standard,
                'properties': {
                    'code_ref': section_number,
                    'test_required': 'listed and labeled' in sentence.lower()
                }
            })
        
        return relationships
    
    def _extract_permissions(self, sentence: str, section_number: str) -> List[Dict]:
        """Extract PERMITTED_IN relationships."""
        relationships = []
        
        # Check if sentence contains permission language
        is_permission = any(re.search(pattern, sentence, re.IGNORECASE) 
                          for pattern in self.PERMISSION_PATTERNS)
        
        if not is_permission:
            return relationships
        
        # Extract equipment and locations
        equipment_terms = self._find_equipment_in_text(sentence)
        location_terms = self._find_locations_in_text(sentence)
        
        for equipment in equipment_terms:
            for location in location_terms:
                relationships.append({
                    'type': 'PERMITTED_IN',
                    'source': equipment,
                    'target': location,
                    'properties': {
                        'code_ref': section_number,
                        'if_condition': self._extract_condition(sentence)
                    }
                })
        
        return relationships
    
    def _extract_device_requirements(self, sentence: str, section_number: str) -> List[Dict]:
        """Extract REQUIRES_DEVICE relationships."""
        relationships = []
        
        # Check for requirement language
        is_requirement = any(re.search(pattern, sentence, re.IGNORECASE) 
                           for pattern in self.REQUIREMENT_PATTERNS)
        
        if not is_requirement:
            return relationships
        
        # Extract equipment and safety devices
        equipment_terms = self._find_equipment_in_text(sentence)
        device_terms = self._find_safety_devices_in_text(sentence)
        
        for equipment in equipment_terms:
            for device in device_terms:
                relationships.append({
                    'type': 'REQUIRES_DEVICE',
                    'source': equipment,
                    'target': device,
                    'properties': {
                        'code_ref': section_number,
                        'mandatory': True
                    }
                })
        
        return relationships
    
    def _find_equipment_in_text(self, text: str) -> List[str]:
        """Find equipment mentions in text and normalize to canonical form."""
        found = []
        doc = self.nlp(text)
        
        # Look for noun chunks that might be equipment
        for chunk in doc.noun_chunks:
            canonical = self.normalizer.normalize_equipment(chunk.text, fuzzy_threshold=0.70)
            if canonical and canonical not in found:
                found.append(canonical)
        
        return found
    
    def _find_locations_in_text(self, text: str) -> List[str]:
        """Find location mentions in text and normalize to canonical form."""
        found = []
        doc = self.nlp(text)
        
        for chunk in doc.noun_chunks:
            canonical = self.normalizer.normalize_location(chunk.text, fuzzy_threshold=0.70)
            if canonical and canonical not in found:
                found.append(canonical)
        
        return found
    
    def _find_materials_in_text(self, text: str) -> List[str]:
        """Find material mentions in text and normalize to canonical form."""
        found = []
        doc = self.nlp(text)
        
        for chunk in doc.noun_chunks:
            canonical = self.normalizer.normalize_material(chunk.text, fuzzy_threshold=0.65)
            if canonical and canonical not in found:
                found.append(canonical)
        
        return found
    
    def _find_safety_devices_in_text(self, text: str) -> List[str]:
        """Find safety device mentions in text and normalize to canonical form."""
        found = []
        doc = self.nlp(text)
        
        for chunk in doc.noun_chunks:
            canonical = self.normalizer.normalize_safety_device(chunk.text, fuzzy_threshold=0.75)
            if canonical and canonical not in found:
                found.append(canonical)
        
        return found
    
    def _extract_reason(self, sentence: str) -> str:
        """Extract reason or explanation from sentence."""
        # Look for common reason indicators
        reason_patterns = [
            r'because\s+(.+?)(?:\.|$)',
            r'due to\s+(.+?)(?:\.|$)',
            r'to prevent\s+(.+?)(?:\.|$)',
            r'to avoid\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in reason_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_condition(self, sentence: str) -> str:
        """Extract conditional clause from sentence."""
        # Look for condition indicators
        condition_patterns = [
            r'(?:if|when|where)\s+(.+?)(?:\.|$)',
            r'provided that\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in condition_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""


    def _extract_material_usage(self, sentence: str, section_number: str) -> List[Dict]:
        """Extract MADE_OF relationships between equipment and materials."""
        relationships = []
        
        # Try each material usage pattern
        for pattern in self.MATERIAL_USAGE_PATTERNS:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                # Extract equipment and material
                equipment_term = match.group(1).strip()
                material_term = match.group(2).strip()
                
                # Normalize
                equipment = self.normalizer.normalize_equipment(equipment_term)
                material = self.normalizer.normalize_material(material_term)
                
                if equipment and material:
                    relationships.append({
                        'type': 'MADE_OF',
                        'source': equipment,
                        'target': material,
                        'properties': {
                            'code_ref': section_number,
                            'requirement': 'mandatory'
                        }
                    })
        
        # Also look for simple "material pipe/duct" patterns
        equipment_terms = self._find_equipment_in_text(sentence)
        material_terms = self._find_materials_in_text(sentence)
        
        # If we find both in same sentence with pipe/duct keywords, link them
        if equipment_terms and material_terms and any(kw in sentence.lower() for kw in ['pipe', 'piping', 'duct', 'ductwork', 'tubing']):
            for equipment in equipment_terms:
                for material in material_terms:
                    # Avoid duplicates
                    exists = any(r['type'] == 'MADE_OF' and r['source'] == equipment and r['target'] == material 
                               for r in relationships)
                    if not exists:
                        relationships.append({
                            'type': 'MADE_OF',
                            'source': equipment,
                            'target': material,
                            'properties': {
                                'code_ref': section_number,
                                'requirement': 'optional'
                            }
                        })
        
        return relationships


# Test the relationship extractor
if __name__ == "__main__":
    extractor = RelationshipExtractor()
    
    # Test sentences from HVAC codes
    test_cases = [
        {
            'text': "Furnaces shall not be installed in sleeping rooms or bathrooms.",
            'section': "303.3"
        },
        {
            'text': "Air handlers shall maintain a minimum clearance of 24 inches from combustible materials.",
            'section': "308.4"
        },
        {
            'text': "Boilers shall comply with ASME Boiler and Pressure Vessel Code.",
            'section': "1004.1"
        },
        {
            'text': "Water heaters shall be equipped with a temperature and pressure relief valve.",
            'section': "1006.3"
        },
        {
            'text': "Gas-fired appliances may be installed in mechanical rooms where provided with adequate ventilation.",
            'section': "303.1"
        },
    ]
    
    print("=== Relationship Extraction Tests ===\n")
    for test in test_cases:
        print(f"Text: {test['text']}")
        print(f"Section: {test['section']}")
        relationships = extractor.extract_relationships(test['text'], test['section'])
        
        if relationships:
            for rel in relationships:
                print(f"  → {rel['type']}: {rel['source']} -> {rel['target']}")
                print(f"     Properties: {rel['properties']}")
        else:
            print("  → No relationships extracted")
        print()
