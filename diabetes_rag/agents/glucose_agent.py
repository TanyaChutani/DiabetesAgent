import numpy as np

def glucose_agent(query):
    try:
        values = list(map(float, query.split()))
        avg = np.mean(values)

        if avg > 180:
            status = "Hyperglycemia"
        elif avg < 70:
            status = "Hypoglycemia"
        else:
            status = "Normal"

        return f"Average: {avg:.2f}, Status: {status}"
    except:
        return "Invalid glucose input"