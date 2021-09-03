import subprocess
import sys
import os
import glob

def has_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
        )
        return True
    except:
        return False

def convert(source_path, target_path):
    subprocess.run(
        ["ffmpeg", "-i", source_path, target_path],
        capture_output=True,
    )


def main(path, from_fmt, to_fmt):
    assert isinstance(path, str)
    glob_str = os.path.join(path, f"*.{from_fmt}")
    source_files = list(sorted(glob.glob(glob_str)))
    if len(source_files) == 0:
        print(f"No .{from_fmt} files were found")
        exit(0)
    for source_file_path in source_files:
        head, tail = os.path.split(source_file_path)
        tail_name, tail_ext = tail.split(".")
        assert tail_ext == from_fmt
        new_path = os.path.join(head, f"{tail_name}.{to_fmt}")
        print(f"Converting {source_file_path} to {new_path}")
        if os.path.isfile(new_path):
            print(f"  Skipping {new_path} because already exists")
            continue
        convert(source_file_path, new_path)
    print("Done")
    

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc not in [2, 3, 4]:
        print(f"Usage: {sys.argv[0]} path/to/files [from_fmt = wav] [to_fmt = flac]")
        exit(0)
    path = sys.argv[1]
    from_fmt = sys.argv[2] if argc >= 3 else "wav"
    to_fmt = sys.argv[3] if argc == 4 else "flac"
    if not os.path.exists(path):
        print(f"\"{path}\" is not a valid path")
        exit(1)
    if not from_fmt.isalnum():
        print(f"{from_fmt} is not a valid file extension")
        exit(1)
    if not to_fmt.isalnum():
        print(f"{to_fmt} is not a valid file extension")
        exit(1)
    if not has_ffmpeg():
        print(f"ffmpeg can't be run. Please make sure ffmpeg is in your PATH environment variable.")
        exit(1)
    main(path, from_fmt, to_fmt)