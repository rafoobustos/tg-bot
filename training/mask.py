import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os

def get_skin_mask(image, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Генерирует маску кожи для входного изображения.
    
    Args:
        image: Может быть numpy array (H,W,3) или PIL Image
        device: Устройство для вычислений ('cuda' или 'cpu')
    
    Returns:
        numpy array: Маска кожи в формате (H,W) с значениями от 0 до 1
    """
    # Конвертируем PIL Image в numpy array если нужно
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Проверяем формат изображения
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Изображение должно быть RGB (H,W,3)")
    
    # Нормализуем изображение в диапазон [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Конвертируем в HSV
    img_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Определяем диапазон цвета кожи в HSV
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Конвертируем в YCrCb
    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # Определяем диапазон цвета кожи в YCrCb
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Объединяем маски
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    
    # Применяем медианный фильтр для удаления шума
    global_mask = cv2.medianBlur(global_mask, 3)
    
    # Морфологические операции для улучшения маски
    kernel = np.ones((4, 4), np.uint8)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, kernel)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, kernel)
    
    # Нормализуем маску в диапазон [0, 1]
    global_mask = global_mask.astype(np.float32) / 255.0
    
    return global_mask

def visualize_mask(image, mask, output_path=None):
    """
    Визуализирует маску кожи на изображении.
    
    Args:
        image: Исходное изображение (numpy array)
        mask: Маска кожи (numpy array)
        output_path: Путь для сохранения результата (опционально)
    
    Returns:
        numpy array: Изображение с наложенной маской
    """
    # Создаем копию изображения
    masked_img = image.copy()
    
    # Применяем маску к каждому каналу
    for c in range(3):
        masked_img[:, :, c] = masked_img[:, :, c] * mask
    
    # Если указан путь для сохранения
    if output_path:
        # Создаем директорию если нужно
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Сохраняем результат
        cv2.imwrite(output_path, cv2.cvtColor((masked_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    return masked_img

def process_image(image_path, output_dir=None):
    """
    Обрабатывает изображение и генерирует маску кожи.
    
    Args:
        image_path: Путь к входному изображению
        output_dir: Директория для сохранения результатов (опционально)
    
    Returns:
        tuple: (маска кожи, изображение с наложенной маской)
    """
    # Загружаем изображение
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path
    
    # Генерируем маску
    mask = get_skin_mask(image)
    
    # Визуализируем результат
    masked_image = visualize_mask(image, mask)
    
    # Если указана директория для сохранения
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем маску
        cv2.imwrite(os.path.join(output_dir, 'mask.png'), (mask * 255).astype(np.uint8))
        
        # Сохраняем результат с наложенной маской
        cv2.imwrite(
            os.path.join(output_dir, 'masked_image.png'),
            cv2.cvtColor((masked_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
    
    return mask, masked_image

if __name__ == "__main__":
    # Пример использования
    import argparse
    
    parser = argparse.ArgumentParser(description='Генерация маски кожи для изображения')
    parser.add_argument('input_path', help='Путь к входному изображению')
    parser.add_argument('--output_dir', help='Директория для сохранения результатов', default='output')
    
    args = parser.parse_args()
    
    # Обрабатываем изображение
    mask, masked_image = process_image(args.input_path, args.output_dir)
    
    print(f"Маска кожи и результат сохранены в директории: {args.output_dir}") 