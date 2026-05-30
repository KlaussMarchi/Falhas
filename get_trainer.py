import json

with open("Model/Analysis.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "class Trainer:" in "".join(cell["source"]):
        print("".join(cell["source"]))

