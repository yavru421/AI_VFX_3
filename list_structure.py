#!/usr/bin/env python3
import os
import sys
import json

def get_tree(path):
    """Recursively build a dictionary representing the directory tree."""
    tree = {"name": os.path.basename(path) or path}
    if os.path.isdir(path):
        tree["type"] = "directory"
        tree["children"] = []
        try:
            with os.scandir(path) as it:
                for entry in it:
                    tree["children"].append(get_tree(entry.path))
        except PermissionError:
            tree["children"].append({"name": "PermissionError", "type": "error"})
    else:
        tree["type"] = "file"
    return tree

if __name__ == '__main__':
    # Use first command-line argument as root directory or default to current directory.
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    directory_tree = get_tree(root)
    output_filename = "directory_structure.json"
    with open(output_filename, "w") as f:
        json.dump(directory_tree, f, indent=4)
    print(f"JSON structure saved to {output_filename}")
