import json

notebook_path = r"notebooks/03_modeling.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, cell in enumerate(data.get("cells", [])):
    if cell.get("cell_type") == "code":
        exec_count = cell.get("execution_count")
        source = "".join(cell.get("source", []))
        print(f"Cell Index: {i}, Execution Count: {exec_count}")
        print("Source Code:")
        print(source)
        print("-" * 50)
