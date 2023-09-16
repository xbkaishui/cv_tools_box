import cv2
import os
from loguru import logger
# 视频文件的路径
video_path = '/Users/xbkaishui/Downloads/1693882450663727.mp4'


# 图像保存目录
output_directory = 'output_images'

# 创建图像保存目录（如果不存在）
os.makedirs(output_directory, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        # 分辨率-宽度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 分辨率-高度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_interval = round((fps / 2 + 0.5))
logger.info(
            "fps {}, width {}, height {} frame_interval {}",
            fps,
            width,
            height,
            frame_interval,
        )
# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 视频帧计数
frame_count = 0

# 读取视频帧
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 保存图像
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_directory, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)

    frame_count += 1

# 释放视频对象
cap.release()

print(f"共保存了 {frame_count} 帧图像")
