import cv2
import numpy as np
import random
# from scipy import ndimage


class DataAugmentor:
    def __init__(self, augment_config=None):
        self.config = augment_config or self.get_default_config()

    def get_default_config(self):
        """获取默认增强配置 - 适配你的训练需求"""
        return {
            # 基础几何变换
            'flip_prob': 0.5,
            'rotate_prob': 0.4,
            'rotate_range': [-15, 15],
            'translate_prob': 0.3,
            'translate_range': [-0.1, 0.1],  # 相对比例

            # 颜色变换
            'brightness_prob': 0.4,
            'brightness_range': [0.7, 1.3],
            'contrast_prob': 0.3,
            'contrast_range': [0.7, 1.3],
            'saturation_prob': 0.3,
            'saturation_range': [0.7, 1.3],

            # 噪声和模糊
            'gaussian_noise_prob': 0.2,
            'gaussian_noise_std': [0, 10],
            'gaussian_blur_prob': 0.2,
            'gaussian_blur_range': [0.5, 1.5],

            # 高级增强
            'cutout_'
            'prob': 0.2,
            'cutout_params': {'num_holes': 1, 'hole_size': [8, 20]},
        }

    def adjust_brightness_contrast(self, image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        """调整亮度和对比度"""
        brightness = random.uniform(brightness_range[0], brightness_range[1])
        contrast = random.uniform(contrast_range[0], contrast_range[1])

        image = image.astype(np.float32)
        image = image * contrast + (brightness - 1) * 128
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def adjust_saturation(self, image, saturation_range=(0.7, 1.3)):
        """调整饱和度"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            saturation_scale = random.uniform(saturation_range[0], saturation_range[1])
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image

    def random_rotate(self, image, angle_range=(-15, 15)):
        """随机旋转"""
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        angle = random.uniform(angle_range[0], angle_range[1])

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101)
        return rotated_image

    def random_translate(self, image, translate_range=(-0.1, 0.1)):
        """随机平移"""
        height, width = image.shape[:2]
        max_dx = translate_range[1] * width
        max_dy = translate_range[1] * height

        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)

        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, translation_matrix, (width, height),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT_101)
        return translated_image

    def random_flip(self, image):
        """随机水平翻转"""
        return cv2.flip(image, 1)

    def add_gaussian_noise(self, image, std_range=(0, 10)):
        """添加高斯噪声"""
        std = random.uniform(std_range[0], std_range[1])
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def add_gaussian_blur(self, image, sigma_range=(0.5, 1.5)):
        """添加高斯模糊"""
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        return blurred

    def random_cutout(self, image, num_holes=1, hole_size=(8, 20)):
        """随机遮挡 - 简化版本"""
        height, width = image.shape[:2]
        image_with_holes = image.copy()

        for _ in range(num_holes):
            hole_w = random.randint(hole_size[0], hole_size[1])
            hole_h = random.randint(hole_size[0], hole_size[1])

            x1 = random.randint(0, width - hole_w)
            y1 = random.randint(0, height - hole_h)
            x2 = x1 + hole_w
            y2 = y1 + hole_h

            # 用均值颜色填充，更自然的遮挡
            if len(image.shape) == 3:
                fill_color = np.mean(image[y1:y2, x1:x2], axis=(0, 1)).astype(np.uint8)
            else:
                fill_color = np.mean(image[y1:y2, x1:x2]).astype(np.uint8)

            cv2.rectangle(image_with_holes, (x1, y1), (x2, y2), fill_color.tolist(), -1)

        return image_with_holes

    def elastic_transform(self, image, alpha=30, sigma=5):
        """弹性变换 - 产生自然形变"""
        random_state = np.random.RandomState(None)
        shape = image.shape[:2]

        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1

        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return cv2.remap(image, indices[1].astype(np.float32), indices[0].astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def augment_single_image(self, image):
        """
        对单张图片进行增强
        注意：输入图像应该是uint8格式，0-255范围
        """
        aug_image = image.copy()

        # 应用各种增强方法（按合理顺序）

        # 1. 首先应用几何变换
        if random.random() < self.config['flip_prob']:
            aug_image = self.random_flip(aug_image)

        if random.random() < self.config['rotate_prob']:
            aug_image = self.random_rotate(aug_image, self.config['rotate_range'])

        if random.random() < self.config['translate_prob']:
            aug_image = self.random_translate(aug_image, self.config['translate_range'])

        # 2. 然后应用颜色变换
        if random.random() < self.config['brightness_prob'] or random.random() < self.config['contrast_prob']:
            aug_image = self.adjust_brightness_contrast(
                aug_image,
                self.config['brightness_range'],
                self.config['contrast_range']
            )

        if random.random() < self.config['saturation_prob']:
            aug_image = self.adjust_saturation(aug_image, self.config['saturation_range'])

        # 3. 最后应用噪声和高级变换
        if random.random() < self.config['gaussian_noise_prob']:
            aug_image = self.add_gaussian_noise(aug_image, self.config['gaussian_noise_std'])

        if random.random() < self.config['gaussian_blur_prob']:
            aug_image = self.add_gaussian_blur(aug_image, self.config['gaussian_blur_range'])

        if random.random() < self.config['cutout_prob']:
            aug_image = self.random_cutout(aug_image, **self.config['cutout_params'])

        return aug_image

    def augment_batch(self, images_batch):
        """
        对批量图片进行增强
        适配你的数据格式：(batch_size, channels, height, width) 或 (batch_size, height, width, channels)
        """
        batch_size = images_batch.shape[0]
        augmented_batch = []

        for i in range(batch_size):
            img = images_batch[i]

            # 处理不同的输入格式
            if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW格式
                # 转换为HWC进行增强
                if img.shape[0] == 3:  # RGB
                    img_hwc = img.transpose(1, 2, 0)
                else:  # 灰度
                    img_hwc = img[0]  # (1,H,W) -> (H,W)
            else:
                img_hwc = img

            # 确保图像是uint8格式
            if img_hwc.max() <= 1.0:
                img_255 = (img_hwc * 255).astype(np.uint8)
            else:
                img_255 = img_hwc.astype(np.uint8)

            # 应用增强
            aug_img = self.augment_single_image(img_255)

            # 转换回原始格式和范围
            if img_hwc.max() <= 1.0:
                aug_img_normalized = aug_img.astype(np.float32) / 255.0
            else:
                aug_img_normalized = aug_img.astype(np.float32)

            # 转换回原始形状
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                if img.shape[0] == 3:  # RGB
                    aug_img_chw = aug_img_normalized.transpose(2, 0, 1)
                else:  # 灰度
                    aug_img_chw = aug_img_normalized[np.newaxis, :, :]  # (H,W) -> (1,H,W)
                augmented_batch.append(aug_img_chw)
            else:
                augmented_batch.append(aug_img_normalized)

        return np.array(augmented_batch)