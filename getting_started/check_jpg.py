from PIL import Image
import matplotlib.pyplot as plt

# 打开原始图像
img = Image.open('/home/Isaac-GR00T/getting_started/remote_images/observation_20250410_033634.jpg')

# 将图像转换为RGB模式（如果尚未）
img = img.convert('RGB')

# 获取图像的像素数据
r, g, b = img.split()

# 交换R和B通道
swapped_img = Image.merge('RGB', (b, g, r))

# 使用matplotlib显示原始图像和处理后的图像
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(swapped_img)
axes[1].set_title('Swapped R and B Channels')
axes[1].axis('off')
# plt.show()
plt.savefig('output_image.png')

