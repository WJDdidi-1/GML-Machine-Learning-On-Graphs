'''
For HMDB51 and UCF101 datasets:

Code extracts frames from video at a rate of 25fps and scales the
larger dimension of the frame to 256 pixels.
After extraction of all frames, write a "done" file to signify completion.

Usage:
  python extract.py <video_dir> <frame_dir> <start_idx> <end_idx>
'''

import sys, os
import subprocess
from tqdm import tqdm
import threading
import ffmpeg

def run_ffmpeg_extract(input_file, outdir, resize_str):
    os.makedirs(outdir, exist_ok=True)
    output_pattern = os.path.join(outdir, 'frame_%04d.jpg')
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_pattern, r=25, vf=f'scale={resize_str}')
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        print(f"[FFmpeg ERROR] {input_file}: {e.stderr.decode()}")

def extract(vid_dir, frame_dir, start, end, redo=False):
    class_list = sorted(os.listdir(vid_dir))[start:end]
    print("Classes =", class_list)
  
    for ic, cls in enumerate(class_list):
        vlist = sorted(os.listdir(os.path.join(vid_dir, cls)))
        print(f'Processing {len(vlist)} videos in class "{cls}"')

        for v in vlist:
            input_file = os.path.join(vid_dir, cls, v)
            outdir = os.path.join(frame_dir, cls, v[:-4])

            if os.path.isfile(os.path.join(outdir, 'done')) and not redo:
                continue

            try:
                os.makedirs(outdir, exist_ok=True)

                # Get video resolution
                probe_cmd = f'ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "{input_file}"'
                o = subprocess.check_output(probe_cmd, shell=True).decode('utf-8')
                lines = o.splitlines()
                width = int(lines[0].split('=')[1])
                height = int(lines[1].split('=')[1])

                resize_str = '-1:256' if width > height else '256:-1'

                # Run ffmpeg extraction
                thread = threading.Thread(target=run_ffmpeg_extract, args=(input_file, outdir, resize_str))
                thread.start()
                thread.join()

                # Mark as done
                open(os.path.join(outdir, 'done'), 'a').close()

            except Exception as e:
                print(f"[ERROR] Class={cls}, Video={v}, {repr(e)}")

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python extract.py <video_dir> <frame_dir> <start_idx> <end_idx>")
        sys.exit(1)

    vid_dir   = sys.argv[1]
    frame_dir = sys.argv[2]
    start     = int(sys.argv[3])
    end       = int(sys.argv[4])
    extract(vid_dir, frame_dir, start, end, redo=True)