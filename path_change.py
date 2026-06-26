import json
from pathlib import Path

tasks_path = Path("/Users/baturu/Documents/Project_CV/tasks.json")

with open(tasks_path, "r") as f:
    tasks = json.load(f)

for task in tasks:
    old_path = task["data"]["image"]
    task["data"]["image"] = old_path.replace(
        "/data/local-files/?d=",
        "/data/local-files/?d=clean_dataset/"
    )

with open(tasks_path, "w") as f:
    json.dump(tasks, f, indent=2)

print("fixed_tasks.json")