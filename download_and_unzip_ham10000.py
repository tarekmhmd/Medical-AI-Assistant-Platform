import os
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile

# المسارات
local_path = "D:/project 2/data/datasets_raw/"
target_directory = "D:/project 2/data/HAM10000_ready/"

# إنشاء المجلدات لو مش موجودة
os.makedirs(local_path, exist_ok=True)
os.makedirs(target_directory, exist_ok=True)

part1_file = os.path.join(local_path, "HAM10000_images_part_1.zip")
part2_file = os.path.join(local_path, "HAM10000_images_part_2.zip")

# URLs للتحميل
part1_url = "https://isic-challenge-data.s3.amazonaws.com/HAM10000_images_part_1.zip"
part2_url = "https://isic-challenge-data.s3.amazonaws.com/HAM10000_images_part_2.zip"

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{os.path.basename(dest_path)} موجودة بالفعل، لن يتم التحميل.")
        return
    print(f"تحميل {os.path.basename(dest_path)} ...")
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(dest_path, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
    print(f"تم تحميل {os.path.basename(dest_path)}.\n")

def unzip_file(zip_path, extract_to):
    folder_name = os.path.splitext(os.path.basename(zip_path))[0]
    dest_folder = os.path.join(extract_to, folder_name)
    if os.path.exists(dest_folder):
        print(f"{folder_name} مفكوكة بالفعل، لن يتم فكها.")
        return
    print(f"فك ضغط {folder_name} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.namelist(), desc=f'Unzipping {folder_name}'):
            zip_ref.extract(member, extract_to)
    print(f"تم فك ضغط {folder_name}.\n")

# تحميل الملفات
download_file(part1_url, part1_file)
download_file(part2_url, part2_file)

# فك الضغط
unzip_file(part1_file, target_directory)
unzip_file(part2_file, target_directory)

# ملخص الصور المفكوكة
print("\nملخص كل الملفات بعد فك الضغط:")
all_files = [str(f) for f in Path(target_directory).rglob('*') if f.is_file()]
for f in all_files:
    print(f)