"""
Canonical vocabulary for HVAC domain entities with synonym mappings.
This enables fuzzy matching between user queries and graph nodes.
"""

from typing import Dict, List, Optional
from difflib import SequenceMatcher


class EntityNormalizer:
    """
    Maps entity variations to canonical forms for consistent graph representation.
    Supports fuzzy matching for user query mapping.
    """
    
    # Equipment canonical names with synonyms
    EQUIPMENT_CANONICAL = {
        "Air Conditioner": ["air conditioner", "ac unit", "cooling unit", "air conditioning unit", "hvac unit"],
        "Air Handler": ["air handler", "ahu", "air handling unit", "handler unit", "air distribution unit"],
        "Boiler": ["boiler", "steam boiler", "hot water boiler", "heating boiler", "water boiler"],
        "Chimney": ["chimney", "flue", "vent stack", "exhaust stack"],
        "Coil": ["coil", "heating coil", "cooling coil", "evaporator coil", "condenser coil"],
        "Compressor": ["compressor", "refrigerant compressor", "ac compressor", "cooling compressor"],
        "Condenser": ["condenser", "condensing unit", "outdoor unit", "condenser unit"],
        "Control": ["control", "control system", "controller", "thermostat", "control device"],
        "Damper": ["damper", "air damper", "fire damper", "smoke damper", "control damper"],
        "Duct": ["duct", "ductwork", "air duct", "ventilation duct", "hvac duct"],
        "Evaporator": ["evaporator", "evaporator unit", "indoor coil", "cooling coil"],
        "Exhaust Fan": ["exhaust fan", "ventilation fan", "vent fan", "extraction fan"],
        "Fan": ["fan", "blower", "air mover", "ventilation fan"],
        "Filter": ["filter", "air filter", "hvac filter", "filtration system"],
        "Flue": ["flue", "flue pipe", "vent pipe", "exhaust pipe"],
        "Furnace": ["furnace", "gas furnace", "heater", "forced air furnace", "heating unit"],
        "Heat Pump": ["heat pump", "heating and cooling system", "reversible heat pump"],
        "Heating Coil": ["heating coil", "heat coil", "warm air coil"],
        "Humidifier": ["humidifier", "humidity control", "moisture control"],
        "Pipe": ["pipe", "piping", "refrigerant pipe", "water pipe"],
        "Pump": ["pump", "circulation pump", "water pump", "refrigerant pump"],
        "Relief Valve": ["relief valve", "pressure relief valve", "safety relief valve", "prv"],
        "Safety Valve": ["safety valve", "safety relief valve", "pressure relief"],
        "Thermostat": ["thermostat", "temperature control", "temp control", "climate control"],
        "Valve": ["valve", "control valve", "shutoff valve", "isolation valve"],
        "Vent": ["vent", "ventilation", "vent system", "exhaust vent"],
        "Ventilator": ["ventilator", "ventilation system", "air exchange system"],
        "Water Heater": ["water heater", "hot water heater", "hwh", "domestic water heater"],
        "Condensing Unit": ["condensing unit", "outdoor condensing unit", "condenser"],
        "Refrigeration System": ["refrigeration system", "cooling system", "refrigerant system"],
        "Pressure Vessel": ["pressure vessel", "vessel", "storage tank"],
        "Heat Exchanger": ["heat exchanger", "coil", "exchanger"],
        "Expansion Tank": ["expansion tank", "expansion vessel", "buffer tank"],
        "Gas Appliance": ["gas appliance", "fuel-burning appliance", "combustion appliance"],
    }
    
    # Location canonical names with synonyms
    LOCATION_CANONICAL = {
        "Attic": ["attic", "attic space", "upper attic", "roof space"],
        "Basement": ["basement", "cellar", "lower level", "underground space"],
        "Bathroom": ["bathroom", "restroom", "washroom", "bath"],
        "Bedroom": ["bedroom", "sleeping room", "bed room", "sleeping area"],
        "Building": ["building", "structure", "premises", "facility"],
        "Ceiling": ["ceiling", "overhead", "ceiling space", "above ceiling"],
        "Closet": ["closet", "storage closet", "utility closet"],
        "Conditioned Space": ["conditioned space", "climate controlled space", "heated space", "cooled space"],
        "Crawl Space": ["crawl space", "crawlspace", "under-floor space"],
        "Dwelling Unit": ["dwelling unit", "apartment", "residential unit", "living unit"],
        "Exterior": ["exterior", "outside", "outdoor", "external"],
        "Floor": ["floor", "floor space", "floor level"],
        "Garage": ["garage", "carport", "vehicle storage"],
        "Indoor": ["indoor", "inside", "interior space", "enclosed space"],
        "Interior": ["interior", "inside", "indoor space"],
        "Kitchen": ["kitchen", "cooking area", "galley"],
        "Living Room": ["living room", "family room", "common area", "lounge"],
        "Mechanical Room": ["mechanical room", "equipment room", "utility room", "machinery room", "boiler room"],
        "Occupied Space": ["occupied space", "habitable space", "living space", "occupied area"],
        "Outdoor": ["outdoor", "outside", "exterior", "open air"],
        "Plenum": ["plenum", "plenum space", "return air plenum", "ceiling plenum"],
        "Return Air": ["return air", "return air path", "return duct"],
        "Roof": ["roof", "rooftop", "roof space", "roof area"],
        "Shaft": ["shaft", "vertical shaft", "service shaft", "utility shaft"],
        "Structure": ["structure", "building", "construction"],
        "Unoccupied Space": ["unoccupied space", "uninhabited space", "non-occupied area"],
        "Wall": ["wall", "partition", "wall assembly"],
    }
    
    # Material canonical names with synonyms
    MATERIAL_CANONICAL = {
        "Aluminum": ["aluminum", "aluminium", "al"],
        "Brass": ["brass", "brass alloy", "copper alloy"],
        "Combustible Material": ["combustible material", "flammable material", "burnable material", "combustibles"],
        "Concrete": ["concrete", "cement", "concrete material"],
        "Copper": ["copper", "cu", "copper material"],
        "CPVC": ["cpvc", "chlorinated pvc", "cpvc pipe"],
        "Drywall": ["drywall", "gypsum board", "sheetrock", "plasterboard"],
        "Fiberglass": ["fiberglass", "fibreglass", "glass fiber", "fiberglass insulation"],
        "Galvanized Steel": ["galvanized steel", "galvanized", "zinc-coated steel"],
        "Gypsum": ["gypsum", "gypsum board", "plaster"],
        "Insulation": ["insulation", "thermal insulation", "insulation material"],
        "Iron": ["iron", "cast iron", "iron material"],
        "Lead": ["lead", "pb", "lead material"],
        "Masonry": ["masonry", "brick", "stone", "masonry material"],
        "Metal": ["metal", "metallic", "metal material"],
        "Noncombustible Material": ["noncombustible material", "non-combustible", "fire-resistant material", "noncombustibles"],
        "Plastic": ["plastic", "plastic material", "polymer"],
        "PVC": ["pvc", "polyvinyl chloride", "pvc pipe"],
        "Stainless Steel": ["stainless steel", "stainless", "ss", "corrosion-resistant steel"],
        "Steel": ["steel", "steel material", "carbon steel"],
        "Wood": ["wood", "lumber", "timber", "wooden material"],
    }
    
    # Standard canonical names with synonyms
    STANDARD_CANONICAL = {
        "ANSI": ["ansi", "american national standards institute"],
        "ASHRAE": ["ashrae", "american society of heating refrigerating and air-conditioning engineers"],
        "ASME": ["asme", "american society of mechanical engineers"],
        "ASSE": ["asse", "american society of sanitary engineering"],
        "ASTM": ["astm", "american society for testing and materials"],
        "CSA": ["csa", "canadian standards association"],
        "ICC": ["icc", "international code council"],
        "International Building Code": ["international building code", "ibc", "building code"],
        "International Fire Code": ["international fire code", "ifc", "fire code"],
        "International Plumbing Code": ["international plumbing code", "ipc", "plumbing code"],
        "NFPA": ["nfpa", "national fire protection association"],
        "NSF": ["nsf", "national sanitation foundation"],
        "SMACNA": ["smacna", "sheet metal and air conditioning contractors national association"],
        "UL": ["ul", "underwriters laboratories"],
        "IIAR": ["iiar", "international institute of ammonia refrigeration"],
        "AHRI": ["ahri", "air-conditioning heating and refrigeration institute"],
    }
    
    # SafetyDevice canonical names with synonyms
    SAFETY_DEVICE_CANONICAL = {
        "Combustion Air": ["combustion air", "combustion air supply", "air intake"],
        "Draft Hood": ["draft hood", "draft diverter", "hood"],
        "Fire Damper": ["fire damper", "fire-rated damper", "fire protection damper"],
        "Pressure Relief Valve": ["pressure relief valve", "prv", "relief valve"],
        "Relief Valve": ["relief valve", "safety relief valve", "pressure relief"],
        "Safety Relief Valve": ["safety relief valve", "safety valve", "srp"],
        "Safety Valve": ["safety valve", "pressure safety valve"],
        "Smoke Damper": ["smoke damper", "smoke control damper"],
        "Temperature Relief Valve": ["temperature relief valve", "t&p valve", "temperature and pressure relief"],
        "Disconnect Switch": ["disconnect switch", "electrical disconnect", "service disconnect", "shutoff switch"],
        "Emergency Shutoff": ["emergency shutoff", "emergency stop", "e-stop", "emergency disconnect"],
        "Low-Water Cutoff": ["low-water cutoff", "lwco", "water level control"],
    }
    
    def __init__(self):
        """Initialize the entity normalizer with reverse lookup dictionaries."""
        # Build reverse lookup: synonym -> canonical
        self.equipment_lookup = self._build_reverse_lookup(self.EQUIPMENT_CANONICAL)
        self.location_lookup = self._build_reverse_lookup(self.LOCATION_CANONICAL)
        self.material_lookup = self._build_reverse_lookup(self.MATERIAL_CANONICAL)
        self.standard_lookup = self._build_reverse_lookup(self.STANDARD_CANONICAL)
        self.safety_device_lookup = self._build_reverse_lookup(self.SAFETY_DEVICE_CANONICAL)
        
    def _build_reverse_lookup(self, canonical_dict: Dict[str, List[str]]) -> Dict[str, str]:
        """Build reverse lookup from synonym to canonical name."""
        reverse = {}
        for canonical, synonyms in canonical_dict.items():
            for synonym in synonyms:
                reverse[synonym.lower()] = canonical
        return reverse
    
    def normalize_equipment(self, text: str, fuzzy_threshold: float = 0.8) -> Optional[str]:
        """Normalize equipment name to canonical form."""
        return self._normalize(text, self.equipment_lookup, self.EQUIPMENT_CANONICAL, fuzzy_threshold)
    
    def normalize_location(self, text: str, fuzzy_threshold: float = 0.8) -> Optional[str]:
        """Normalize location name to canonical form."""
        return self._normalize(text, self.location_lookup, self.LOCATION_CANONICAL, fuzzy_threshold)
    
    def normalize_material(self, text: str, fuzzy_threshold: float = 0.8) -> Optional[str]:
        """Normalize material name to canonical form."""
        return self._normalize(text, self.material_lookup, self.MATERIAL_CANONICAL, fuzzy_threshold)
    
    def normalize_standard(self, text: str, fuzzy_threshold: float = 0.8) -> Optional[str]:
        """Normalize standard name to canonical form."""
        return self._normalize(text, self.standard_lookup, self.STANDARD_CANONICAL, fuzzy_threshold)
    
    def normalize_safety_device(self, text: str, fuzzy_threshold: float = 0.8) -> Optional[str]:
        """Normalize safety device name to canonical form."""
        return self._normalize(text, self.safety_device_lookup, self.SAFETY_DEVICE_CANONICAL, fuzzy_threshold)
    
    def _normalize(self, text: str, lookup: Dict[str, str], canonical_dict: Dict[str, List[str]], 
                   fuzzy_threshold: float) -> Optional[str]:
        """
        Normalize text to canonical form using exact match or fuzzy matching.
        
        Args:
            text: Input text to normalize
            lookup: Reverse lookup dictionary (synonym -> canonical)
            canonical_dict: Forward dictionary (canonical -> synonyms)
            fuzzy_threshold: Minimum similarity score (0.0-1.0) for fuzzy match
            
        Returns:
            Canonical name if match found, None otherwise
        """
        text_lower = text.lower().strip()
        
        # Exact match
        if text_lower in lookup:
            return lookup[text_lower]
        
        # Fuzzy match
        best_match = None
        best_score = 0.0
        
        for canonical, synonyms in canonical_dict.items():
            for synonym in synonyms:
                score = SequenceMatcher(None, text_lower, synonym.lower()).ratio()
                if score > best_score and score >= fuzzy_threshold:
                    best_score = score
                    best_match = canonical
        
        return best_match
    
    def get_equipment_synonyms(self, canonical: str) -> List[str]:
        """Get all synonyms for a canonical equipment name."""
        return self.EQUIPMENT_CANONICAL.get(canonical, [])
    
    def get_location_synonyms(self, canonical: str) -> List[str]:
        """Get all synonyms for a canonical location name."""
        return self.LOCATION_CANONICAL.get(canonical, [])
    
    def get_material_synonyms(self, canonical: str) -> List[str]:
        """Get all synonyms for a canonical material name."""
        return self.MATERIAL_CANONICAL.get(canonical, [])
    
    def get_standard_synonyms(self, canonical: str) -> List[str]:
        """Get all synonyms for a canonical standard name."""
        return self.STANDARD_CANONICAL.get(canonical, [])
    
    def get_safety_device_synonyms(self, canonical: str) -> List[str]:
        """Get all synonyms for a canonical safety device name."""
        return self.SAFETY_DEVICE_CANONICAL.get(canonical, [])


# Singleton instance
entity_normalizer = EntityNormalizer()


# Test the normalizer
if __name__ == "__main__":
    normalizer = EntityNormalizer()
    
    # Test equipment normalization
    print("=== Equipment Normalization Tests ===")
    test_equipment = ["air handler", "AHU", "heater", "gas furnace", "AC unit"]
    for term in test_equipment:
        canonical = normalizer.normalize_equipment(term)
        print(f"{term:20s} -> {canonical}")
    
    print("\n=== Location Normalization Tests ===")
    test_locations = ["mechanical room", "boiler room", "attic space", "garage"]
    for term in test_locations:
        canonical = normalizer.normalize_location(term)
        print(f"{term:20s} -> {canonical}")
    
    print("\n=== Material Normalization Tests ===")
    test_materials = ["combustible material", "wood", "stainless steel", "concrete"]
    for term in test_materials:
        canonical = normalizer.normalize_material(term)
        print(f"{term:20s} -> {canonical}")
    
    print("\n=== Standard Normalization Tests ===")
    test_standards = ["UL", "NFPA", "ASHRAE", "building code"]
    for term in test_standards:
        canonical = normalizer.normalize_standard(term)
        print(f"{term:20s} -> {canonical}")
    
    print("\n=== Fuzzy Matching Tests ===")
    fuzzy_tests = ["air handleer", "furnce", "mechnical room", "combustable"]
    for term in fuzzy_tests:
        eq_match = normalizer.normalize_equipment(term, fuzzy_threshold=0.75)
        loc_match = normalizer.normalize_location(term, fuzzy_threshold=0.75)
        mat_match = normalizer.normalize_material(term, fuzzy_threshold=0.75)
        result = eq_match or loc_match or mat_match or "No match"
        print(f"{term:20s} -> {result}")
