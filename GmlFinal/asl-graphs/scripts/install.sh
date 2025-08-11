python ./scripts/install.py
git clone https://github.com/whwu95/MVFNet.git
python MVFNet/data_process/video_resize.py {org_path} {resized_dir} --level 2 -se 256
python MVFNet/data_process/video2image.py {resized_dir} {frame_dir} --level 2 --lib opencv --prefix image_%04d.jpg
rm -r {resized_dir}
rm -r ./MVFNet