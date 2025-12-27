import cv2
import numpy as np
import random

class CutMix:
    """CutMix数据增强"""

    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, images, labels):
        if np.random.random() > self.prob:
            return images, labels

        batch_size = images.shape[0]

        # 随机打乱批次顺序
        indices = np.random.permutation(batch_size)

        # 生成lambda参数
        lam = np.random.beta(self.alpha, self.alpha)

        # 获取图像尺寸 (N, C, H, W)
        H, W = images.shape[2], images.shape[3]

        # 生成裁剪区域
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)

        # 随机选择裁剪中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 计算裁剪区域边界
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # 确保裁剪区域有效
        if x2 <= x1 or y2 <= y1:
            return images, labels

        # 创建混合图像
        mixed_images = images.copy()

        # 应用CutMix (N, C, H, W) 格式
        mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]

        # 计算混合比例并调整标签
        area = (x2 - x1) * (y2 - y1)
        lam = 1. - area / (H * W)

        mixed_labels = labels * lam + labels[indices] * (1. - lam)#软标签（不使用）

        return mixed_images, mixed_labels


class DataAugmentor:
    def __init__(self, use_cutmix=True, cutmix_alpha=1.0, cutmix_prob=0.5):
        self.use_cutmix = use_cutmix
        if use_cutmix:
            self.cutmix = CutMix(alpha=cutmix_alpha, prob=cutmix_prob)

        # 初始化增强配置
        self.config = self.get_default_config()

    def get_default_config(self):
        """获取默认增强配置"""
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
            'cutout_prob': 0.2,
            'cutout_params': {'num_holes': 1, 'hole_size': [8, 20]},
        }

    def apply_cutmix(self, images, labels):
        """应用CutMix增强"""
        if self.use_cutmix and len(images) > 1:  # 需要至少2张图片
            return self.cutmix(images, labels)
        return images, labels

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
        """随机遮挡"""
        height, width = image.shape[:2]
        image_with_holes = image.copy()

        for _ in range(num_holes):
            hole_w = random.randint(hole_size[0], hole_size[1])
            hole_h = random.randint(hole_size[0], hole_size[1])

            x1 = random.randint(0, width - hole_w)
            y1 = random.randint(0, height - hole_h)
            x2 = x1 + hole_w
            y2 = y1 + hole_h

            # 用均值颜色填充
            if len(image.shape) == 3:
                fill_color = np.mean(image[y1:y2, x1:x2], axis=(0, 1)).astype(np.uint8)
            else:
                fill_color = np.mean(image[y1:y2, x1:x2]).astype(np.uint8)

            cv2.rectangle(image_with_holes, (x1, y1), (x2, y2), fill_color.tolist(), -1)

        return image_with_holes

    def augment_single_image(self, image):
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