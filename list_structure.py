#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from datetime import datetime

def print_tree(path, file, prefix="", is_last=True):
    """Print directory tree in a compact format and write to file."""
    name = os.path.basename(path) or path
    
    # Create the line prefix with proper branching characters
    marker = "└── " if is_last else "├── "
    line = f"{prefix}{marker}{name}\n"
    file.write(line)
    
    # If it's a directory, process its contents
    if os.path.isdir(path):
        try:
            entries = sorted([
                entry for entry in os.scandir(path) 
                if not entry.name.startswith('.')
            ], key=lambda x: x.name.lower())
            
            for i, entry in enumerate(entries):
                is_last_entry = (i == len(entries) - 1)
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(entry.path, file, next_prefix, is_last_entry)
        except PermissionError:
            file.write(f"{prefix}    └── <Permission Denied>\n")

def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"directory_structure_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Directory Structure for: {root}\n")
        f.write("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("="*50 + "\n\n")
        
        print_tree(root, f)
        
        f.write("\nDone!")
    
    print(f"\nDirectory structure saved to: {output_file}")

if __name__ == '__main__':
    main()
