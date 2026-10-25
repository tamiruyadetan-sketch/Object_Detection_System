
from ultralytics import YOLO
from pathlib import Path
from collections import Counter

model = YOLO("yolo11n.pt")

image_folder = Path("dataset/raw_images")
output_folder = Path("runs/detect/all_images")
output_folder.mkdir(parents=True, exist_ok=True)

images = [
    p for p in image_folder.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

total_counts = Counter()

print(f"Total images: {len(images)}")

for image_path in images:
    results = model(str(image_path), verbose=False)

    counts = Counter()

    for box in results[0].boxes:
        name = results[0].names[int(box.cls[0])]
        counts[name] += 1
        total_counts[name] += 1

    # Save image with bounding boxes
    annotated = results[0].plot()
    import cv2
    cv2.imwrite(
        str(output_folder / image_path.name),
        annotated
    )

    print(f"{image_path.name}: {dict(counts)}")

print("\n===== TOTAL DETECTED OBJECTS =====")
for name, count in total_counts.items():
    print(f"{name}: {count}")

print(f"\nAnnotated images saved to: {output_folder}")