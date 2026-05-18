class RiskAnalyzer:
    """Calculates dynamically weighted risk scores based on aggregated environmental metrics."""
    
    def __init__(self, db_manager=None):
        self.db = db_manager

    def calculate_risk(self, crowd_density: float, incident_frequency: float, loitering_events: float):
        """
        Calculates risk score based on:
        (crowd_density * 0.3 + incident_frequency * 0.4 + loitering_events * 0.3)
        Inputs should be normalized 0-100.
        """
        risk_score = (crowd_density * 0.3) + (incident_frequency * 0.4) + (loitering_events * 0.3)
        risk_score = min(100.0, max(0.0, risk_score))
        
        if risk_score <= 30:
            classification = "LOW"
        elif risk_score <= 60:
            classification = "MEDIUM"
        else:
            classification = "HIGH"
            
        return risk_score, classification
