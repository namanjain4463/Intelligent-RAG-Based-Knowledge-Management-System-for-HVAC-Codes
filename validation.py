"""
Input Validation Module for HVAC GraphRAG System
Validates and sanitizes user inputs before processing
"""
import re
from typing import Tuple


def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validate user query before processing.
    
    Args:
        query: User's natural language query
    
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    # Check if query is empty or whitespace only
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    # Check query length
    if len(query) > 2000:
        return False, f"Query too long (max 2000 characters, got {len(query)})"
    
    if len(query) < 3:
        return False, "Query too short (minimum 3 characters)"
    
    # Check for suspicious patterns (basic security)
    suspicious_patterns = [
        r'\bDROP\b',
        r'\bDELETE\b',
        r'\bCREATE\b',
        r'\bMERGE\b',
        r'<script>',
        r'javascript:',
        r'\x00',  # Null bytes
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "Query contains forbidden patterns"
    
    # Check for excessive special characters (may indicate injection attempt)
    special_char_count = len(re.findall(r'[{}()\[\]<>]', query))
    if special_char_count > 20:
        return False, "Query contains too many special characters"
    
    return True, ""


def sanitize_query(query: str) -> str:
    """
    Sanitize user query by removing potentially harmful content.
    
    Args:
        query: User's natural language query
    
    Returns:
        Sanitized query string
    """
    # Remove null bytes
    query = query.replace('\x00', '')
    
    # Normalize whitespace
    query = ' '.join(query.split())
    
    # Remove HTML tags if any
    query = re.sub(r'<[^>]+>', '', query)
    
    # Limit consecutive special characters
    query = re.sub(r'([^\w\s])\1{2,}', r'\1\1', query)
    
    return query.strip()


def validate_section_number(section: str) -> bool:
    """
    Validate section number format.
    
    Args:
        section: Section number (e.g., "901", "901.1", "901.1.2")
    
    Returns:
        True if valid format, False otherwise
    """
    # Valid pattern: digits with optional decimal sub-sections
    pattern = r'^\d{1,4}(\.\d+)*$'
    return bool(re.match(pattern, section))


def validate_entity_name(name: str) -> Tuple[bool, str]:
    """
    Validate entity name (equipment, location, etc.)
    
    Args:
        name: Entity name
    
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not name or not name.strip():
        return False, "Entity name cannot be empty"
    
    if len(name) > 100:
        return False, "Entity name too long (max 100 characters)"
    
    # Entity names should be alphanumeric with spaces, hyphens, underscores
    if not re.match(r'^[\w\s\-]+$', name):
        return False, "Entity name contains invalid characters"
    
    return True, ""
