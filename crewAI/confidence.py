import statistics

def calculate_confidence(criteria_results, chief_adjusted: bool):
    bands = [c["band"] for c in criteria_results]

    mean = statistics.mean(bands)
    stdev = statistics.pstdev(bands)

    agreement = max(0, 1 - stdev / 1.5)
    justification_lengths = [len(c.get("justification", "")) for c in criteria_results]
    evidence = min(1, statistics.mean(justification_lengths)/350)
    adjustment_score = 0.6 if chief_adjusted else 1.0
    spread = max(bands) - min(bands)
    spread_score = max(0, 1 - spread/2)

    confidence = agreement*0.4 + evidence*0.25 + adjustment_score*0.2 + spread_score*0.15
    confidence_pct = round(confidence*100,1)

    if confidence_pct >= 85:
        label = "Very High"
    elif confidence_pct >= 70:
        label = "High"
    elif confidence_pct >= 55:
        label = "Medium"
    else:
        label = "Low"

    return confidence_pct, label
