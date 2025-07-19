from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import os
import uuid
import numpy as np
import pytesseract

app = Flask(__name__)

# Statik klasörler
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# YOLOv8 modelini yükle (.pt dosyanın yolu)
model = YOLO(
    r"C:\Users\Beyza\Workspace\gyk-computer-vision\plate-project\runs\detect\train2\weights\best.pt"
)


def draw_bounding_boxes(image, results, threshold=0.5):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = box.conf.item()
            if conf < threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            text = f"{conf:.2f}"
            draw.text((x1, y1 - 20), text, fill="red", font=font)

    return draw_image


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="Lütfen bir dosya seçin!")

        unique_id = str(uuid.uuid4())
        original_filename = f"{unique_id}_original.jpg"
        result_filename = f"{unique_id}_result.jpg"

        original_image = Image.open(io.BytesIO(file.read())).convert("RGB")
        original_image.save(f"static/uploads/{original_filename}")

        # YOLOv8 modeline uygun numpy array'e çevir
        img_np = np.array(original_image)

        # Modeli çalıştır
        results = model(img_np)

        # Görsele bounding box çiz
        result_image = draw_bounding_boxes(original_image, results, threshold=0.5)
        result_image.save(f"static/results/{result_filename}")

        # Tespit edilen plaka sayısı ve OCR ile plaka metinleri
        detections = 0
        plate_texts = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = box.conf.item()
                if conf > 0.5:
                    detections += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # OCR için kırp
                    cropped_plate = original_image.crop((x1, y1, x2, y2))
                    text = pytesseract.image_to_string(
                        cropped_plate, lang="eng", config="--psm 7"
                    )
                    plate_texts.append(text.strip())

        return render_template(
            "result.html",
            original_image=f"uploads/{original_filename}",
            result_image=f"results/{result_filename}",
            detections=detections,
            plate_texts=plate_texts,
        )

    except Exception as e:
        return render_template("index.html", error=f"Hata oluştu: {str(e)}")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
