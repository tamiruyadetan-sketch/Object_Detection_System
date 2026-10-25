"""Local web UI for YOLO image and video detection.

Run with: .\\.venv\\Scripts\\python.exe app.py
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import base64
import mimetypes
import platform
import shutil
import threading
import uuid
from datetime import datetime
from collections import Counter
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
RUNS = ROOT / "runs" / "web_ui"
RUNS.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = RUNS / "settings.json"
RESULTS_FILE = RUNS / "results.json"
RESULTS_LOCK = threading.Lock()
LIVE_LOCK = threading.Lock()
LIVE: dict | None = None
DEFAULT_SETTINGS = {
    "confidence": 0.50, "iou": 0.45, "image_size": 640,
    "classes": ["person", "car", "bicycle", "motorcycle", "truck"],
    "model_file": "yolo11n.pt", "model_name": "YOLO11 Nano", "model_version": "11n",
    "device": "cpu", "output_folder": "web_ui", "show_boxes": True,
    "show_labels": True, "show_ids": False,
}
SETTINGS = DEFAULT_SETTINGS.copy()
if SETTINGS_FILE.exists():
    try:
        SETTINGS.update(json.loads(SETTINGS_FILE.read_text()))
    except (OSError, json.JSONDecodeError):
        pass
MODEL = YOLO(str(ROOT / SETTINGS["model_file"]))
CLASS_IDS = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "truck": 7}


def parse_upload(handler: SimpleHTTPRequestHandler) -> tuple[str, bytes]:
    """Read the single file sent by FormData without an extra web framework."""
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length)
    content_type = handler.headers.get("Content-Type", "")
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    for part in message.iter_attachments():
        filename = part.get_filename()
        if filename:
            return Path(filename).name, part.get_payload(decode=True)
    raise ValueError("No file was included in the upload.")


def count_objects(result) -> Counter:
    return Counter(result.names[int(box.cls[0])] for box in result.boxes)


def confidence_summary(result) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        grouped.setdefault(name, []).append(float(box.conf[0]))
    return {name: round(sum(values) / len(values), 3) for name, values in grouped.items()}


def stamp_frame(frame):
    """Add the local processing date and time to saved image/video results."""
    text = datetime.now().strftime("%B %d, %Y  %I:%M:%S %p")
    cv2.rectangle(frame, (10, 10), (355, 43), (6, 27, 58), -1)
    cv2.putText(frame, text, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def read_results() -> list[dict]:
    with RESULTS_LOCK:
        try:
            return json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
        except (OSError, json.JSONDecodeError):
            return []


def write_results(records: list[dict]) -> None:
    with RESULTS_LOCK:
        RESULTS_FILE.write_text(json.dumps(records, indent=2))


def prediction_args() -> dict:
    return {"conf": float(SETTINGS["confidence"]), "iou": float(SETTINGS["iou"]),
            "imgsz": int(SETTINGS["image_size"]), "device": "0" if SETTINGS["device"] == "gpu" else "cpu",
            "classes": [CLASS_IDS[name] for name in SETTINGS["classes"] if name in CLASS_IDS] or None}


def output_directory() -> Path:
    """Keep user selected outputs safely inside the project's runs directory."""
    name = Path(str(SETTINGS["output_folder"])).name or "web_ui"
    folder = ROOT / "runs" / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def detect_image(source: Path, job: Path) -> tuple[Path, Counter, dict[str, float]]:
    result = MODEL(str(source), verbose=False, **prediction_args())[0]
    output = job / f"detected_{source.stem}.jpg"
    cv2.imwrite(str(output), stamp_frame(result.plot()))
    return output, count_objects(result), confidence_summary(result)


def detect_video(source: Path, job: Path) -> tuple[Path, Counter, dict[str, float]]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("The video could not be opened.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = job / f"detected_{source.stem}.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    counts: Counter = Counter()
    confidences: dict[str, list[float]] = {}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = MODEL(frame, verbose=False, **prediction_args())[0]
            counts.update(count_objects(result))
            for name, score in confidence_summary(result).items():
                confidences.setdefault(name, []).append(score)
            writer.write(stamp_frame(result.plot()))
    finally:
        capture.release()
        writer.release()
    return output, counts, {name: round(sum(scores) / len(scores), 3) for name, scores in confidences.items()}


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def end_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/api/live/stats":
            with LIVE_LOCK:
                if not LIVE: self.end_json({"status": "stopped"}); return
                self.end_json({"status": LIVE["status"], "fps": LIVE["fps"], "active": LIVE["active"],
                               "counts": dict(LIVE["unique_counts"]), "tracking": "ByteTrack", "model": SETTINGS["model_name"]})
            return
        if self.path == "/api/live/stream":
            self.end_json({"message": "Frames are streamed from the browser camera to /api/live/frame and returned annotated."})
            return
        if self.path == "/api/results":
            self.end_json({"results": read_results()})
            return
        if self.path == "/api/stats":
            records = read_results()
            totals: Counter = Counter()
            for record in records:
                totals.update(record.get("counts", {}))
            self.end_json({"sessions": len(records), "objects_by_class": dict(totals),
                           "recent": records[:5]})
            return
        if self.path.startswith("/api/results/"):
            pieces = self.path.split("/")
            record_id = pieces[3] if len(pieces) > 3 else ""
            record = next((item for item in read_results() if item["id"] == record_id), None)
            if not record:
                self.end_json({"error": "Result not found."}, 404)
                return
            if len(pieces) > 4 and pieces[4] == "download":
                target = RUNS / Path(record["result_file"]).name
                if not target.exists():
                    self.end_json({"error": "Saved file not found."}, 404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
                self.send_header("Content-Length", str(target.stat().st_size))
                self.end_headers()
                with target.open("rb") as file: shutil.copyfileobj(file, self.wfile)
                return
            self.end_json(record)
            return
        if self.path == "/api/settings":
            self.end_json({**SETTINGS, "model_loaded": MODEL is not None,
                           "gpu_available": bool(cv2.cuda.getCudaEnabledDeviceCount()),
                           "platform": platform.system()})
            return
        if self.path.startswith("/results/"):
            target = RUNS / Path(self.path.removeprefix("/results/")).name
            if not target.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(target.stat().st_size))
            self.end_headers()
            with target.open("rb") as file:
                shutil.copyfileobj(file, self.wfile)
            return
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        global LIVE, MODEL
        if self.path == "/api/live/start":
            with LIVE_LOCK:
                if LIVE and LIVE.get("writer"): LIVE["writer"].release()
                LIVE = {"id": uuid.uuid4().hex, "status": "running", "unique_ids": set(), "unique_counts": Counter(),
                        "active": 0, "fps": 0, "last_time": None, "writer": None, "recording": False,
                        "last_frame": None, "frames": 0}
            self.end_json({"id": LIVE["id"], "status": "running", "tracker": "ByteTrack"}); return
        if self.path == "/api/live/frame":
            with LIVE_LOCK:
                if not LIVE or LIVE["status"] != "running": self.end_json({"error": "No active live session."}, 409); return
            try:
                payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                frame = cv2.imdecode(__import__("numpy").frombuffer(payload, dtype="uint8"), cv2.IMREAD_COLOR)
                if frame is None: raise ValueError("Invalid camera frame.")
                import time
                try:
                    result = MODEL.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, **prediction_args())[0]
                except Exception:
                    # ByteTrack needs the optional ``lap`` package.  Detection must remain usable
                    # when that tracker dependency is not installed.
                    result = MODEL(frame, verbose=False, **prediction_args())[0]
                annotated = result.plot()
                with LIVE_LOCK:
                    now = time.perf_counter(); previous = LIVE["last_time"]; LIVE["fps"] = round(1 / (now - previous), 1) if previous else 0; LIVE["last_time"] = now
                    LIVE["active"] = len(result.boxes)
                    if result.boxes.id is not None:
                        for box, track_id in zip(result.boxes, result.boxes.id.int().tolist()):
                            key = (result.names[int(box.cls[0])], track_id)
                            if key not in LIVE["unique_ids"]: LIVE["unique_ids"].add(key); LIVE["unique_counts"][key[0]] += 1
                    if LIVE["recording"]:
                        if LIVE["writer"] is None:
                            h, w = annotated.shape[:2]; LIVE["writer"] = cv2.VideoWriter(str(RUNS / f"live_{LIVE['id']}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                        LIVE["writer"].write(annotated)
                    LIVE["last_frame"] = annotated; LIVE["frames"] += 1
                    ok, encoded = cv2.imencode(".jpg", annotated)
                    self.end_json({"image": base64.b64encode(encoded.tobytes()).decode(), "fps": LIVE["fps"], "active": LIVE["active"], "counts": dict(LIVE["unique_counts"])})
            except Exception as error: self.end_json({"error": str(error)}, 400)
            return
        if self.path == "/api/live/recording/start":
            with LIVE_LOCK:
                if not LIVE: self.end_json({"error":"Start the camera first."},409); return
                LIVE["recording"] = True
            self.end_json({"recording": True}); return
        if self.path == "/api/live/recording/stop":
            with LIVE_LOCK:
                if not LIVE: self.end_json({"error":"No active session."},409); return
                LIVE["recording"] = False
                if LIVE["writer"]: LIVE["writer"].release(); LIVE["writer"] = None
            self.end_json({"recording": False}); return
        if self.path == "/api/live/snapshot":
            with LIVE_LOCK:
                if not LIVE or LIVE["last_frame"] is None: self.end_json({"error":"No frame available."},409); return
                name = f"snapshot_{LIVE['id']}.jpg"; cv2.imwrite(str(RUNS / name), LIVE["last_frame"])
            self.end_json({"url": f"/results/{name}"}); return
        if self.path in ("/api/live/stop", "/api/live/save-session"):
            with LIVE_LOCK:
                if not LIVE: self.end_json({"error":"No active session."},409); return
                LIVE["status"] = "stopped"
                if LIVE["writer"]: LIVE["writer"].release(); LIVE["writer"] = None
                if self.path.endswith("save-session"):
                    video = f"live_{LIVE['id']}.mp4"; exists = (RUNS / video).exists()
                    record = {"id": uuid.uuid4().hex, "filename": f"camera_session_{LIVE['id'][:8]}", "type": "Live Camera", "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"), "status":"Completed", "total":sum(LIVE["unique_counts"].values()), "counts":dict(LIVE["unique_counts"]), "confidences":{}, "tracking_ids":[track for _,track in LIVE["unique_ids"]], "result_file":video if exists else "", "original_file":video if exists else "", "job_folder":""}
                    records=read_results();records.insert(0,record);write_results(records)
                response={"status":"stopped", "counts":dict(LIVE["unique_counts"]), "saved": self.path.endswith("save-session")}
            self.end_json(response); return
        if self.path == "/api/settings":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                incoming = json.loads(self.rfile.read(length))
                updated = DEFAULT_SETTINGS.copy()
                updated.update({key: incoming[key] for key in updated if key in incoming})
                updated["confidence"] = min(1, max(0.01, float(updated["confidence"])))
                updated["iou"] = min(1, max(0.01, float(updated["iou"])))
                updated["image_size"] = int(updated["image_size"])
                updated["classes"] = [item for item in updated["classes"] if item in CLASS_IDS]
                updated["device"] = "gpu" if updated["device"] == "gpu" else "cpu"
                model_path = ROOT / Path(updated["model_file"]).name
                if not model_path.is_file() or model_path.suffix != ".pt":
                    raise ValueError("Model file was not found in the project folder.")
                if updated["model_file"] != SETTINGS["model_file"]:
                    MODEL = YOLO(str(model_path))
                SETTINGS.update(updated)
                SETTINGS_FILE.write_text(json.dumps(SETTINGS, indent=2))
                self.end_json({**SETTINGS, "model_loaded": True})
            except Exception as error:
                self.end_json({"error": str(error)}, 400)
            return
        if self.path != "/api/detect":
            self.send_error(404)
            return
        try:
            filename, content = parse_upload(self)
            extension = Path(filename).suffix.lower()
            image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            if extension not in image_extensions | video_extensions:
                raise ValueError("Please upload an image or video file.")
            job = output_directory() / uuid.uuid4().hex
            job.mkdir()
            source = job / f"upload{extension}"
            source.write_bytes(content)
            output, counts, confidences = (detect_image(source, job) if extension in image_extensions else detect_video(source, job))
            public_name = f"{job.name}_{output.name}"
            published = RUNS / public_name
            shutil.copy2(output, published)
            original_name = f"{job.name}_original{extension}"
            shutil.copy2(source, RUNS / original_name)
            processed_at = datetime.now().astimezone().isoformat(timespec="seconds")
            is_camera_session = filename.startswith("camera_session_")
            record = {"id": uuid.uuid4().hex, "filename": filename,
                      "type": "Live Camera" if is_camera_session else ("Image" if extension in image_extensions else "Video"),
                      "created_at": processed_at, "status": "Completed",
                      "total": sum(counts.values()), "counts": dict(counts), "confidences": confidences,
                      "tracking_ids": [], "result_file": public_name, "original_file": original_name,
                      "job_folder": str(job.relative_to(ROOT))}
            records = read_results(); records.insert(0, record); write_results(records)
            self.end_json({
                "kind": "image" if extension in image_extensions else "video",
                "url": f"/results/{public_name}",
                "counts": dict(counts),
                "total": sum(counts.values()),
                "filename": filename,
                "id": record["id"], "original_url": f"/results/{original_name}", "confidences": confidences,
                "processed_at": processed_at,
            })
        except Exception as error:
            self.end_json({"error": str(error)}, 400)

    def do_DELETE(self):
        if not self.path.startswith("/api/results/"):
            self.end_json({"error": "Not found."}, 404); return
        record_id = self.path.rsplit("/", 1)[-1]
        records = read_results()
        record = next((item for item in records if item["id"] == record_id), None)
        if not record:
            self.end_json({"error": "Result not found."}, 404); return
        for filename in (record.get("result_file"), record.get("original_file")):
            path = RUNS / Path(filename).name
            if path.exists(): path.unlink()
        job = ROOT / record.get("job_folder", "")
        if job.is_dir() and job.resolve().is_relative_to(ROOT / "runs"):
            shutil.rmtree(job)
        write_results([item for item in records if item["id"] != record_id])
        self.end_json({"deleted": record_id})

    def log_message(self, format, *args):
        print("[web]", format % args)


if __name__ == "__main__":
    print("AI Object Detection System running at http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), AppHandler).serve_forever()
