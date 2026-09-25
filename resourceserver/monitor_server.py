from fastapi import FastAPI, Request
from threading import Thread, Event
import time, docker, os
import uvicorn

app = FastAPI()
stop_event = Event()
log_thread = None

image_prefixes = ["pandeymanish"]

client = docker.from_env()

def monitor_docker(log_file_path):
    with open(log_file_path, "a") as f:
        while not stop_event.is_set():
            try:
                containers = client.containers.list()
                timestamp_sec = int(time.time())
                timestamp_ms = int(timestamp_sec * 1000)
                for c in containers:
                    try:
                        image_tags = c.image.tags
                        if image_tags:
                            image_name = image_tags[0]
                            if any(image_name.startswith(prefix) for prefix in image_prefixes):
                                line = f"{timestamp_ms},{c.name},{image_name},{c.status}"
                                f.write(line + "\n")
                    except Exception as e:
                        # Log the error and continue
                        print(f"[WARN] Could not read info for container: {getattr(c, 'name', 'unknown')}, error: {e}")
                        continue
                f.flush()
            except Exception as e:
                print(f"[ERROR] Could not list containers or major error: {e}")
            time.sleep(1)


@app.post("/start-recording")
async def start(request: Request):
    global log_thread, stop_event

    data = await request.json()
    folder = data.get("folder", "").strip()
    if not folder:
        return {"error": "Missing 'folder' in request"}
    
    full_path = os.path.join("..", folder)
    os.makedirs(full_path, exist_ok=True)
    log_file_path = os.path.join(full_path, "docker_log.csv")

    # Stop old logging thread if running
    if log_thread is not None and log_thread.is_alive():
        stop_event.set()
        log_thread.join()  # Wait for the old thread to finish

    # Start a new logging thread for the new file
    stop_event.clear()
    log_thread = Thread(target=monitor_docker, args=(log_file_path,))
    log_thread.start()
    return {"status": "Recording started", "file": log_file_path}

@app.post("/stop-recording")
def stop():
    stop_event.set()
    return {"status": "Recording stopped"}

if __name__ == "__main__":
    uvicorn.run("monitor_server:app", host="0.0.0.0", port=8001, reload=False)
