import os
from pathlib import Path
import time
from loguru import logger


def filter_files_by_creation_time(directory, before_threshold_timestamp, after_threshold_timestamp):
    filtered_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            creation_time = os.path.getmtime(file_path)
            if creation_time > before_threshold_timestamp and creation_time < after_threshold_timestamp:
                # logger.info("file_path {} c_time {}", file_path, creation_time)
                file_name = Path(file_path).name
                if file_name == '2.bmp':
                    filtered_files.append(file_path)
    return filtered_files

directory_path = '/Volumes/EXTERNAL_USB/重庆现场测试/storage_back'  # 替换为要搜索的目录路径
threshold_date = '2023-06-06'  # 过滤阈值日期
before_threshold_timestamp = time.mktime(time.strptime(threshold_date, '%Y-%m-%d'))
after_threshold_timestamp = time.mktime(time.strptime('2023-06-07', '%Y-%m-%d'))

filtered_files = filter_files_by_creation_time(directory_path, before_threshold_timestamp, after_threshold_timestamp)
# for file in filtered_files:
#     print(file)

# copy filtered_files to dest dir
logger.info("copy filtered_files to dest dir")
dest_dir = "/tmp/outs"
idx = 1
import shutil
for file in  filtered_files:
    logger.info("copy file {} to {}", file, f'{dest_dir}/{idx}.bmp')
    shutil.copy(file, f'{dest_dir}/{idx}.bmp')
    idx = idx + 1 
