def interpret_risk(score: float):
    if score >= 0.7:
        return "High Risk"
    elif score >= 0.4:
        return "Moderate Risk"
    else:
        return "Low Risk"
