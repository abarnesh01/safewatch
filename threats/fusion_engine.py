class FusionEngine:
    """Aggregates intelligence layers into a unified Threat Score."""
    
    def calculate_threat(self, behavior_score: float, face_score: float, risk_score: float, location_score: float):
        """
        Weights:
        - behavior_score (40%): Driven by ActionClassifier (Fight, Fall)
        - face_score (25%): Driven by Watchlist (Blacklist = High, VIP = Low)
        - risk_score (20%): Driven by RiskAnalyzer (Density, History)
        - location_score (15%): Driven by ZoneManager (Restricted Areas)
        """
        # Normalize inputs to 0-100 scale
        final_score = (behavior_score * 0.40) + (face_score * 0.25) + (risk_score * 0.20) + (location_score * 0.15)
        
        # Apply strict policy overrides
        if face_score > 90: # e.g. Confirmed Blacklist
            final_score = max(final_score, 85.0)
            
        if final_score <= 35:
            return final_score, "LOW"
        elif final_score <= 70:
            return final_score, "MEDIUM"
        else:
            return final_score, "CRITICAL"

    def determine_face_score(self, identity: dict) -> float:
        """Converts identity metadata into a threat score contribution."""
        category = identity.get("category", "UNKNOWN")
        if category == "Blacklist":
            return 100.0
        elif category == "UNKNOWN":
            return 50.0
        elif category == "Employee":
            return 10.0
        elif category == "VIP":
            return 0.0
        return 30.0
