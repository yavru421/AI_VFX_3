import os
import json
import datetime

def get_output_dir():
    # Load output_dir from config.json; default to "output" if not defined
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config.get("output_dir", "output")
    except Exception as e:
        print(f"Error reading config.json: {e}")
        return "output"

def is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

def main():
    output_dir = get_output_dir()
    if not os.path.isdir(output_dir):
        print(f"Output directory '{output_dir}' does not exist.")
        return

    # List immediate subdirectories in the output directory
    subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
               if os.path.isdir(os.path.join(output_dir, d))]
    
    if not subdirs:
        print(f"No subdirectories found in '{output_dir}'.")
        return

    print("Selected images from each output subdirectory:\n")
    for subdir in subdirs:
        # Look for image files in the current subdirectory
        image_files = [f for f in os.listdir(subdir)
                       if os.path.isfile(os.path.join(subdir, f)) and is_image(f)]
        if image_files:
            image_files.sort()  # pick the first image alphabetically
            selected_image = image_files[0]
            image_path = os.path.join(subdir, selected_image)
            # Optionally, show file size and last modified date
            file_size = os.path.getsize(image_path)
            mod_time = os.path.getmtime(image_path)
            mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Directory: {subdir}")
            print(f"  Selected image: {selected_image}")
            print(f"  Size: {file_size} bytes | Last modified: {mod_time_str}\n")
        else:
            print(f"Directory: {subdir} has no image files.\n")

if __name__ == "__main__":
    main()
