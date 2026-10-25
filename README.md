# 🚗 AI-Based Object Detection, Classification and Counting System

An AI-powered computer vision application designed to detect, classify, track, and count objects in images, videos, and live camera streams using YOLO, OpenCV, and object tracking technology.

The system focuses on detecting five main object classes: **Cars, People, Bicycles, Motorcycles, and Trucks**.

## 📌 Project Overview

The AI-Based Object Detection System aims to make object detection and traffic monitoring easier through an intuitive user interface and automated computer vision techniques.

Users can upload images and videos, run object detection, monitor objects through a live camera, and review previously processed results through the history page.

## 🎯 Objectives

- Detect objects automatically using a YOLO-based model.
- Classify detected objects into predefined categories.
- Track objects across video frames.
- Count unique tracked objects to reduce duplicate counting.
- Support image, video, and live camera detection.
- Store and review previous detection results.
- Provide a user-friendly dashboard for monitoring results.

## ✨ Key Features

### 🏠 1. Home Dashboard
- Overview of detection activities.
- Summary of detected object classes.
- Object count statistics.
- Quick access to the main application features.

### 🖼️ 2. Image Detection
- Upload an image for analysis.
- Detect and classify objects automatically.
- Display bounding boxes, class labels, and confidence scores.
- View and download the processed image.

### 🎥 3. Video Detection
- Upload a video for object detection.
- Process video frames using YOLO.
- Track objects across frames.
- Display detection results and object counts.
- Save the processed video when supported.

### 📹 4. Live Camera
- Detect objects from a connected webcam.
- Display real-time detection results.
- Show object labels, bounding boxes, and tracking IDs.
- Monitor object counts during a camera session.
- Save snapshots or recordings when supported.

### 📚 5. Detection History
- Review previous detection sessions.
- View processed images and videos.
- Search and filter saved results when supported.
- Review object counts and detection details.
- Download or delete saved results when supported.

### ⚙️ 6. Settings
- Configure detection confidence thresholds.
- Select supported object classes.
- Configure available model and camera options.
- Manage supported output preferences.

## 🚦 Supported Object Classes

| No. | Object Class | Description |
|---|---|---|
| 1 | Car | Detects cars and other supported car categories |
| 2 | Person | Detects people |
| 3 | Bicycle | Detects bicycles |
| 4 | Motorcycle | Detects motorcycles |
| 5 | Truck | Detects trucks |

**Note:** The current pretrained YOLO model may recognize additional classes. Restricting detection to these five classes depends on the application's configuration.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application and AI processing |
| YOLO (Ultralytics) | Object detection and classification |
| OpenCV | Image and video processing |
| ByteTrack | Object tracking across video frames |
| React | Frontend interface, if configured |
| FastAPI | Backend API, if configured |
| SQLite / PostgreSQL | Detection history storage, if configured |
| Git and GitHub | Version control and project hosting |

*The technologies marked as optional or conditional should only be listed as implemented if they are actually integrated into the current version.*

## 🏗️ System Architecture

The planned system follows this workflow:

1. The user uploads an image or video, or starts the live camera.
2. The application sends image or video frames to the detection pipeline.
3. YOLO identifies objects and assigns class labels and confidence scores.
4. An object tracker associates detections across video frames.
5. The counting component calculates object counts.
6. The application displays the results.
7. Detection results are saved to history when storage functionality is implemented and enabled.

## 📂 Project Structure

The project may be organized as follows:

```text
Object_Detection_System/
│
├── dataset/
│   ├── raw_images/
│   └── videos/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── package.json
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── api/
│       ├── services/
│       └── database/
│
├── models/
│   └── best.pt
│
├── runs/
│   └── detect/
│
├── test_yolo.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `frontend`, `backend`, `models`, and `requirements.txt` entries depend on your actual project structure. Remove or adjust entries that do not exist in your repository.

## 💻 Installation and Setup

### Prerequisites

Install the following tools:

- Python 3.10 or a compatible version supported by your dependencies.
- Visual Studio Code or another Python IDE.
- Git.
- Node.js and npm, if the frontend uses React.
- A webcam, if you want to use live camera detection.

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd Object_Detection_System
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Python Dependencies

```bash
pip install ultralytics opencv-python
```

Install any additional dependencies required by your application.

### Step 4: Prepare the Dataset

Place your test images inside:

```text
dataset/raw_images/
```

Place your test videos inside:

```text
dataset/videos/
```

Create these directories if they do not already exist.

### Step 5: Run Image Detection

Run the image detection script:

```bash
python test_yolo.py
```

The script processes the images in `dataset/raw_images/` and saves annotated images in `runs/detect/all_images/`, based on the current script configuration.

## 📊 Example Output

The system can display results in the following format:

```text
Image: road.jpg

Detected Objects:
Car: 4
Person: 3
Bicycle: 2
Motorcycle: 1
Truck: 2
```

These values are illustrative examples, not measured results from a particular test image.

## 🔢 Object Tracking and Counting

Object tracking associates detections across video frames using tracking IDs.

For example, a car may appear in many consecutive frames. Instead of counting every frame-level detection as a new car, the system can count a tracked ID once per session.

Tracking-based counting may still produce errors if objects are temporarily hidden, leave and re-enter the scene, or receive new tracking IDs.

## 🔒 Privacy and Responsible Use

- Obtain appropriate permission before recording people.
- Protect stored images and videos.
- Avoid collecting or sharing personal information unnecessarily.
- Use the application in accordance with applicable laws and institutional policies.

## 🚀 Future Improvements

- Train a custom YOLO model using a labeled dataset.
- Improve tracking and unique-object counting accuracy.
- Add vehicle counting at virtual traffic lines.
- Implement a complete REST API.
- Integrate the frontend with the backend.
- Add persistent detection history and analytics.
- Export reports in CSV and JSON formats.
- Improve performance for real-time camera processing.
- Add automated testing and deployment.

## 🧪 Testing

The project can be evaluated using:

- Image detection accuracy.
- Object classification performance.
- Video tracking consistency.
- Unique counting accuracy.
- Camera processing speed (FPS).
- Error handling and application stability.

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Submit a pull request.

## 👨‍💻 Author

**Yadetan Tamiru**

Computer Science Student  
Dilla University, Ethiopia

## 📄 License

Choose an appropriate open-source license before distributing the project. For example, you may use the MIT License if its terms fit your project.

---

**AI-Based Object Detection, Classification and Counting System**  
*Smart Vision • Safer World*
