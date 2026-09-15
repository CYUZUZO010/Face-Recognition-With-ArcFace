# Face Recognition with ArcFace ONNX and 5-Point Alignment

A local, camera-based face-recognition pipeline using:

- OpenCV for camera access and image processing
- Haar detection with MediaPipe FaceMesh landmarks
- Five-point face alignment to `112 x 112`
- An ArcFace-style ONNX model for face embeddings
- A local NumPy/JSON face database for enrollment and recognition

## Requirements

- Windows, macOS, or Linux
- Python 3.10-3.13
- A working webcam
- The ArcFace ONNX model at:

```text
models/embedder_arcface.onnx
```

The model is not included in this repository. Add a compatible ArcFace ONNX model at that path before running the embedding, enrollment, evaluation, or recognition scripts.

## Setup

From the project root, create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

(On macOS/Linux, activate with `source .venv/bin/activate` instead, then run the same `pip install` command.)

The `.venv` directory is local to your machine and should not be committed. It is already excluded by `.gitignore`.

## Usage

Run commands from the project root while the virtual environment is active.

### Check the camera

```powershell
python -m src.camera
```

### Test face detection

```powershell
python -m src.detect
```

### Preview five-point alignment

```powershell
python -m src.align
```

Press `q` to quit or `s` to save the current aligned face under `data/debug_aligned/`.

### Enroll a person

```powershell
python -m src.enroll
```

Enter a name when prompted and follow the on-screen capture instructions. Enrollment creates:

```text
data/db/face_db.npz
data/db/face_db.json
data/enroll/<person-name>/
```

### Recognize enrolled people

```powershell
python -m src.recognize
```

The recognition window supports:

- `q`: quit
- `r`: reload the face database
- `+` / `-`: adjust the match threshold
- `d`: toggle debug overlays

### Evaluate the enrolled database

```powershell
python -m src.evaluate
```

## Project layout

```text
face-recognition-5pt/
├── data/
│   ├── db/              Generated face database files
│   ├── debug_aligned/   Saved alignment previews
│   └── enroll/          Captured enrollment faces
├── models/              Local ONNX model files
├── src/
│   ├── align.py         Five-point alignment demo
│   ├── detect.py        Face detection demo
│   ├── embed.py         ArcFace embedding demo
│   ├── enroll.py        Build the local face database
│   ├── evaluate.py      Evaluate enrolled samples
│   ├── haar_5pt.py      Haar and five-point detection pipeline
│   └── recognize.py     Real-time recognition
└── README.md
```

## Troubleshooting

### Camera does not open

Close other applications using the webcam. The scripts use camera index `0` in the main recognition and enrollment flows. If your camera uses another index, update the `cv2.VideoCapture(...)` value in the relevant script.

### MediaPipe cannot be imported

Confirm that the virtual environment is active, then reinstall the pinned version:

```powershell
python -m pip install --force-reinstall mediapipe==0.10.21
```

### The ONNX model cannot be found

Run the commands from the repository root and confirm this file exists:

```text
models/embedder_arcface.onnx
```
