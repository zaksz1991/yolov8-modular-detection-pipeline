import json

def export_to_json(results, output_filename='results.json'):
    structured_data = []
    for r in results:
        for box in r.boxes:
            data = {
                "class_id": int(box.cls),
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0] # [xmin, ymin, xmax, ymax]
            }
            structured_data.append(data)
    
    with open(output_filename, 'w') as f:
        json.dump(structured_data, f, indent=4)
    print(f"Results saved to {output_filename}")