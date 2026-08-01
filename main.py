import os
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO


# ==============================
# APP SETUP
# ==============================

app = FastAPI(
    title="SmartVision AI",
    description="AI Powered Image Recognition System",
    version="1.0"
)


# ==============================
# CORS CONFIGURATION
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# FOLDERS
# ==============================

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ==============================
# STATIC RESULT IMAGES
# ==============================

app.mount(
    "/results",
    StaticFiles(directory=RESULT_FOLDER),
    name="results"
)


# ==============================
# LOAD YOLO MODEL
# ==============================

model = YOLO("yolov8n.pt")


# ==============================
# HOME API
# ==============================

@app.get("/")
def home():
    return {
        "message": "SmartVision AI Backend is Running"
    }


# ==============================
# IMAGE UPLOAD & AI DETECTION
# ==============================

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):

    # ------------------------------
    # Save Uploaded Image
    # ------------------------------

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )


    # ------------------------------
    # Run YOLO Detection
    # ------------------------------

    results = model(upload_path)


    # ------------------------------
    # Detection Data
    # ------------------------------

    detections = []

    object_counts = {}

    result_filename = "result_" + file.filename


    # ------------------------------
    # Process Detection Results
    # ------------------------------

    for result in results:

        # ------------------------------
        # Save Detection Image
        # ------------------------------

        result_path = os.path.join(
            RESULT_FOLDER,
            result_filename
        )

        result.save(
            filename=result_path
        )


        # ------------------------------
        # Process Every Detected Object
        # ------------------------------

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = model.names[
                class_id
            ]


            # ------------------------------
            # Add Detection Information
            # ------------------------------

            detections.append({
                "object": class_name,
                "confidence": round(
                    confidence * 100,
                    2
                )
            })


            # ------------------------------
            # Count Objects
            # ------------------------------

            if class_name in object_counts:

                object_counts[
                    class_name
                ] += 1

            else:

                object_counts[
                    class_name
                ] = 1


    # ------------------------------
    # Send Response to Frontend
    # ------------------------------

    return JSONResponse(
        content={

            "message":
                "Image analyzed successfully",

            "filename":
                file.filename,

            "detections":
                detections,

            "object_counts":
                object_counts,

            "result_image":
                "/results/" + result_filename
        }
    )