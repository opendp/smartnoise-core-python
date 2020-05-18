import json
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.join(script_dir, "..", "whitenoise-core", "validator-rust", "prototypes", "components")
for idx, filename in enumerate(sorted(os.listdir(components_dir))):
    if not filename.endswith(".json"):
        continue

    component_path = os.path.join(components_dir, filename)

    with open(component_path, "r") as component_file:
        component = json.load(component_file)

    component['proto_id'] = idx

    with open(component_path, 'w') as component_file:
        json.dump(component, component_file, indent=2)
