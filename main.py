import os
import shutil
import uuid

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
# CORS
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
# STATIC RESULTS
# ==============================

app.mount(
    "/results",
    StaticFiles(directory=RESULT_FOLDER),
    name="results"
)


# ==============================
# LOAD LIGHTWEIGHT YOLO MODEL
# ==============================

model = YOLO("yolov8n.pt")


# ==============================
# HOME
# ==============================

@app.get("/")
def home():
    return {
        "message": "SmartVision AI Backend is Running"
    }


# ==============================
# IMAGE UPLOAD & DETECTION
# ==============================

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):

    # ------------------------------
    # Create unique filename
    # ------------------------------

    file_extension = os.path.splitext(file.filename)[1]

    unique_id = str(uuid.uuid4())[:8]

    safe_filename = f"image_{unique_id}{file_extension}"

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        safe_filename
    )

    result_filename = f"result_{unique_id}.jpg"

    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )


    # ------------------------------
    # Save uploaded image
    # ------------------------------

    with open(upload_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    # ------------------------------
    # Run YOLO Detection
    # ------------------------------

    results = model.predict(
        source=upload_path,
        imgsz=320,
        conf=0.25,
        device="cpu",
        verbose=False
    )


    # ------------------------------
    # Detection Data
    # ------------------------------

    detections = []

    object_counts = {}


    # ------------------------------
    # Process Results
    # ------------------------------

    for result in results:

        # Save detected image

        result.save(
            filename=result_path
        )


        # Process detected objects

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


            detections.append({

                "object":
                    class_name,

                "confidence":
                    round(
                        confidence * 100,
                        2
                    )
            })


            # Count objects

            object_counts[class_name] = (
                object_counts.get(
                    class_name,
                    0
                ) + 1
            )


    # ------------------------------
    # Delete uploaded image
    # ------------------------------

    try:

        os.remove(
            upload_path
        )

    except Exception:

        pass


    # ------------------------------
    # Response
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