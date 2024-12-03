import cv2
import os
from tqdm import tqdm

def create_video_from_images(image_folder, output_video_path, fps=30):
    # 获取图片文件列表并排序
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])

    # 确保有图片存在
    if not images:
        print("No images found in the folder.")
        return

    # 读取第一张图片来获取帧的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 将每张图片按顺序写入视频
    for image_name in tqdm(images):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        video.write(image)

    # 释放视频写入对象
    video.release()
    print(f"Video saved at {output_video_path}")

# 使用示例
image_folder = 'results2/'  # 替换为你的图片文件夹路径
output_video_path = 'output_yolov11_deepsort.mp4'      # 设置输出视频的路径
create_video_from_images(image_folder, output_video_path, fps=10)  # 设置所需的帧率
