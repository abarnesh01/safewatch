import time
from loguru import logger

class LoiteringDetector:
    """Detects when a person remains in a specific zone longer than a threshold."""
    
    def __init__(self, config: dict):
        self.enabled = config.get("threats", {}).get("loitering", {}).get("enabled", True)
        self.max_dwell_time = config.get("threats", {}).get("loitering", {}).get("max_dwell_time", 60)
        # Map: person_id -> (entry_time, zone_name)
        self.zone_tracking = {} 

    def _get_zone(self, person, zones):
        """Checks if a person's center point is inside any zone."""
        # This requires ZoneManager logic. Assuming we receive zone info or coordinate check.
        # For this skeleton, we assume `zones` is a dict of {zone_name: polygon} and we have a point_in_poly check.
        # Alternatively, if ZoneManager already tags persons, we can use that.
        return person.get_zone() if hasattr(person, 'get_zone') else None

    def detect(self, persons, zones):
        if not self.enabled:
            return []
            
        alerts = []
        now = time.time()
        
        for person in persons:
            in_zone = self._get_zone(person, zones)
            
            if in_zone:
                if person.id not in self.zone_tracking:
                    self.zone_tracking[person.id] = (now, in_zone)
                else:
                    entry_time, z_name = self.zone_tracking[person.id]
                    dwell = now - entry_time
                    
                    if z_name == in_zone:
                        if dwell > self.max_dwell_time:
                            alerts.append({
                                "threat_type": "LOITERING",
                                "severity": "MEDIUM",
                                "confidence": 0.9,
                                "description": f"Person {person.id} loitering in {in_zone} for {int(dwell)}s",
                                "persons_involved": [person.id]
                            })
                    else:
                        # Changed zones
                        self.zone_tracking[person.id] = (now, in_zone)
            else:
                self.zone_tracking.pop(person.id, None)
                
        return alerts
