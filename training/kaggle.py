# Импорт необходимых библиотек
import os
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import time
import zipfile
import cv2
from pathlib import Path
import shutil
import torch.nn.functional as F
# import torch_xla
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.parallel_loader as pl

try:
    import face_alignment
    has_face_alignment = True
except ImportError:
    has_face_alignment = False

# Обновляем пути
FFHQ_PATH = "/kaggle/input/flickrfaceshq-dataset-ffhq"
OUTPUT_DIR = "/kaggle/working/tuned-models"
PROCESSED_DATA_DIR = "/kaggle/working/training_data"

def extract_processed_data():
    """Распаковка предобработанных данных."""
    print("Распаковка предобработанных данных...")
    
    extract_path = '/kaggle/working/training_data'
    os.makedirs(extract_path, exist_ok=True)
    
    # Create required directory structure
    os.makedirs(os.path.join(extract_path, 'train', 'bw'), exist_ok=True)
    os.makedirs(os.path.join(extract_path, 'train', 'color'), exist_ok=True)
    os.makedirs(os.path.join(extract_path, 'val', 'bw'), exist_ok=True)
    os.makedirs(os.path.join(extract_path, 'val', 'color'), exist_ok=True)
    
    with zipfile.ZipFile(PROCESSED_DATA_PATH, 'r') as zip_ref:
        # List all files in the zip
        files = zip_ref.namelist()
        
        # Extract and organize files
        for file in files:
            # Determine if file is for training or validation (you might need to adjust this logic)
            is_train = 'train' in file.lower()
            target_dir = 'train' if is_train else 'val'
            
            # Determine if file is BW or color (you might need to adjust this logic)
            is_bw = 'bw' in file.lower()
            img_type = 'bw' if is_bw else 'color'
            
            # Extract to appropriate directory
            target_path = os.path.join(extract_path, target_dir, img_type)
            zip_ref.extract(file, target_path)
    
    print("Данные распакованы в:", extract_path)
    return extract_path

def num_features_model(m):
    """Возвращает количество features в backbone модели."""
    if hasattr(m, 'fc'):
        return m.fc.in_features
    elif hasattr(m, 'classifier'):
        return m.classifier.in_features
    elif hasattr(m, 'head'):
        return m.head.in_features
    else:
        return 512  # Значение по умолчанию для ResNet34

class SelfAttention(nn.Module):
    """Механизм Self-Attention как в DeOldify"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Получаем Q, K, V
        query = self.query(x).view(batch_size, -1, H*W)
        key = self.key(x).view(batch_size, -1, H*W)
        value = self.value(x).view(batch_size, -1, H*W)
        
        # Вычисляем attention scores
        attention = F.softmax(torch.bmm(query.permute(0,2,1), key), dim=2)
        
        # Применяем attention к value
        out = torch.bmm(value, attention.permute(0,2,1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    """Улучшенный ResidualBlock с нормализацией"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class UpsampleBlock(nn.Module):
    """Улучшенный UpsampleBlock с SubPixel конволюцией"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class NoiseLayer(nn.Module):
    """Добавляет случайный шум для стабилизации обучения"""
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.noise = None
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x)
            return x + self.weight * noise
        return x

class FacialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels//8, 1)
        self.key = nn.Conv2d(channels, channels//8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.ones(1))  # Инициализируем с 1, чтобы сразу дать приоритет
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H*W)
        key = self.key(x).view(batch_size, -1, H*W)
        value = self.value(x).view(batch_size, -1, H*W)
        
        attention = F.softmax(torch.bmm(query.permute(0,2,1), key), dim=2)
        out = torch.bmm(value, attention.permute(0,2,1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class UNetGeneratorModule(nn.Module):
    def __init__(self, arch, n_classes=3):
        super().__init__()
        # Энкодер на базе предобученной модели
        backbone = arch(pretrained=True)
        
        # Анализируем и выводим структуру backbone только при инициализации модели
        print("Backbone structure:", list(backbone.children()))
        
        # Извлекаем слои для энкодера с более точным контролем размеров
        # Основанные на документации структуры ResNet
        self.enc1 = nn.Sequential(*list(backbone.children())[:4])  # 64x128x128 после conv1, bn1, relu, maxpool
        self.enc2 = nn.Sequential(*list(backbone.children())[4])   # 64x64x64 после layer1
        self.enc3 = nn.Sequential(*list(backbone.children())[5])   # 128x32x32 после layer2
        self.enc4 = nn.Sequential(*list(backbone.children())[6])   # 256x16x16 после layer3
        self.enc5 = nn.Sequential(*list(backbone.children())[7])   # 512x8x8 после layer4
        
        # Средний блок с self-attention
        self.middle = nn.Sequential(
            ResidualBlock(512),
            SelfAttention(512),
            ResidualBlock(512)
        )
        
        # Декодеры с U-Net skip соединениями учитывая правильные размеры
        self.dec5 = UNetUpBlock(512, 256)                 # 256x16x16
        self.dec4 = UNetUpBlock(256 + 256, 128)           # 128x32x32
        self.dec3 = UNetUpBlock(128 + 128, 64)            # 64x64x64
        self.dec2 = UNetUpBlock(64 + 64, 32)              # 32x128x128
        
        # Изменяем последний декодер для учета правильного размера входящих тензоров
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResidualBlock(16)
        )
        
        # Финальный слой
        self.final = nn.Sequential(
            nn.Conv2d(16, n_classes, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Убираем все печати размеров для чистого вывода во время обучения
        
        # Энкодер с сохранением выходов для skip-соединений
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Средний блок
        middle = self.middle(e5)
        
        # Декодер с использованием skip-соединений и интерполяцией при необходимости
        d5 = self.dec5(middle)
        
        # Проверяем и подгоняем размеры перед конкатенацией
        if d5.shape[2:] != e4.shape[2:]:
            d5 = F.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        
        if d4.shape[2:] != e3.shape[2:]:
            d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        
        # Используем только d2 для последнего декодера, убираем конкатенацию с e1
        d1 = self.dec1(d2)
        
        # Финальный выход
        out = self.final(d1)
        
        # Интерполируем до нужного размера, если необходимо
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
            
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            ResidualBlock(out_channels)
        )
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.bw_dir = os.path.join(root_dir, 'bw')
        self.color_dir = os.path.join(root_dir, 'color')
        self.transform = transform
        self.images = [f for f in os.listdir(self.bw_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        bw_path = os.path.join(self.bw_dir, img_name)
        color_path = os.path.join(self.color_dir, img_name)
        
        # Загружаем numpy массивы
        bw_image = np.load(bw_path)
        color_image = np.load(color_path)
        
        # Преобразуем в тензоры PyTorch
        bw_image = torch.from_numpy(bw_image).float().permute(2, 0, 1)
        color_image = torch.from_numpy(color_image).float().permute(2, 0, 1)
        
        # Применяем дополнительные преобразования, если они есть
        if self.transform:
            bw_image = self.transform(bw_image)
            color_image = self.transform(color_image)
        
        return {'bw': bw_image, 'color': color_image}

def get_dataloaders(data_dir, batch_size, img_size):
    """Simplified dataloader without augmentation to reduce GPU memory usage"""
    
    # Use the original simple dataset class without augmentation
    train_dataset = ImageDataset(os.path.join(data_dir, 'train'), None)  # No transform
    val_dataset = ImageDataset(os.path.join(data_dir, 'val'), None)  # No transform
    
    # Create data loaders with minimal workers to reduce memory usage
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced number of workers
        pin_memory=False  # Disable pin_memory to save GPU memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=1  # Minimal workers for validation
    )
    
    return train_loader, val_loader

class FacialPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Используем FaceNet или другую модель, специализированную на лицах
        self.face_model = models.resnet18(pretrained=True)
        # Удаляем классификационный слой
        self.face_model = nn.Sequential(*list(self.face_model.children())[:-2])
        # Замораживаем веса
        for param in self.face_model.parameters():
            param.requires_grad = False
        self.mse = nn.MSELoss()
        
    def forward(self, output, target):
        with torch.no_grad():
            output_features = self.face_model(output)
            target_features = self.face_model(target)
        return self.mse(output_features, target_features)

class CombinedLoss(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.vgg = vgg
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
    def get_improved_skin_mask(self, image):
        """Получает маску кожи с использованием улучшенного алгоритма"""
        # Конвертируем тензор в numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
            if image.min() < 0:
                image = (image + 1) / 2
        
        # Используем функцию из skin_mask.py
        from skin_mask import get_skin_mask
        mask = get_skin_mask(image)
        
        # Конвертируем обратно в тензор
        mask = torch.from_numpy(mask).float()
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        # Перемещаем на нужное устройство
        mask = mask.to(image.device if isinstance(image, torch.Tensor) else 'cpu')
        
        return mask

    def compute_smoothness_loss(self, x, mask):
        """Compute smoothness loss with the mask from mask.py"""
        # Compute gradients in x and y directions
        grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        
        # Apply mask to gradients
        mask_x = mask[:, :, :, :-1]
        mask_y = mask[:, :, :-1, :]
        
        # Calculate weighted sum of gradients
        smoothness_loss = (
            torch.mean(grad_x * mask_x) +
            torch.mean(grad_y * mask_y)
        )
        
        return smoothness_loss

    def compute_skin_color_loss(self, output, target, skin_mask):
        """Специализированная функция потерь для кожи с защитой от NaN"""
        # Переводим в LAB цветовое пространство для лучшего сравнения цветов кожи
        output_lab = rgb_to_lab(output)
        target_lab = rgb_to_lab(target)
        
        # Проверка, есть ли пиксели кожи
        if torch.sum(skin_mask) < 1.0:
            # Если нет пикселей кожи, возвращаем нулевую потерю
            return torch.tensor(0.0, device=output.device)
        
        # Добавляем очистку от возможных NaN
        output_lab = torch.nan_to_num(output_lab, nan=0.0, posinf=1.0, neginf=-1.0)
        target_lab = torch.nan_to_num(target_lab, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Разделяем каналы для более точного контроля
        # Повышаем важность a и b каналов (отвечают за цвет) для кожи
        l_loss = F.smooth_l1_loss(output_lab[:, 0:1] * skin_mask, target_lab[:, 0:1] * skin_mask)
        a_loss = F.smooth_l1_loss(output_lab[:, 1:2] * skin_mask, target_lab[:, 1:2] * skin_mask) * 3.0
        b_loss = F.smooth_l1_loss(output_lab[:, 2:3] * skin_mask, target_lab[:, 2:3] * skin_mask) * 3.0
        
        # Добавляем штраф за низкую насыщенность кожи с безопасным вычислением
        epsilon = 1e-8  # Маленькое значение для предотвращения проблем с sqrt(0)
        target_a_b = torch.clamp(target_lab[:, 1:], min=-1.0, max=1.0)
        output_a_b = torch.clamp(output_lab[:, 1:], min=-1.0, max=1.0)
        
        skin_saturation_target = torch.sqrt(target_a_b[:, 0:1]**2 + target_a_b[:, 1:2]**2 + epsilon) * skin_mask
        skin_saturation_output = torch.sqrt(output_a_b[:, 0:1]**2 + output_a_b[:, 1:2]**2 + epsilon) * skin_mask
        
        # Поскольку уже защитили от деления на ноль и NaN, безопасно используем L1
        saturation_loss = F.l1_loss(skin_saturation_output, skin_saturation_target) * 2.0
        
        # Финальная защита от любых NaN в потерях
        l_loss = torch.nan_to_num(l_loss, nan=0.0)
        a_loss = torch.nan_to_num(a_loss, nan=0.0)
        b_loss = torch.nan_to_num(b_loss, nan=0.0)
        saturation_loss = torch.nan_to_num(saturation_loss, nan=0.0)
        
        return l_loss * 0.5 + a_loss + b_loss + saturation_loss

    def forward(self, output, target):
        # Защита входных данных от NaN
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Base components
        l1_loss = self.l1(output, target)
        
        with torch.no_grad():
            vgg_output = self.vgg(output)
            vgg_target = self.vgg(target)
        perceptual_loss = self.mse(vgg_output, vgg_target)
        
        # Get skin mask using the approach from mask.py
        skin_mask = self.get_improved_skin_mask(target)
        
        # Convert to LAB
        output_lab = rgb_to_lab(output)
        target_lab = rgb_to_lab(target)
        
        # Skin area losses
        skin_pixels = skin_mask.sum()
        if skin_pixels > 0:
            # Increase weights for skin losses
            skin_color_loss = self.compute_skin_color_loss(output, target, skin_mask)
            
            # Improved smoothness loss
            skin_smoothness_loss = (
                self.compute_smoothness_loss(output_lab[:, 1:], skin_mask) * 5.0
            )
        else:
            skin_color_loss = torch.tensor(0.0, device=output.device)
            skin_smoothness_loss = torch.tensor(0.0, device=output.device)
        
        # General color loss (с защитой от NaN)
        color_loss = self.mse(
            torch.nan_to_num(output_lab[:, 1:], nan=0.0),
            torch.nan_to_num(target_lab[:, 1:], nan=0.0)
        )
        
        # Защита всех потерь от NaN
        l1_loss = torch.nan_to_num(l1_loss, nan=0.0)
        perceptual_loss = torch.nan_to_num(perceptual_loss, nan=0.0)
        color_loss = torch.nan_to_num(color_loss, nan=0.0)
        skin_color_loss = torch.nan_to_num(skin_color_loss, nan=0.0)
        skin_smoothness_loss = torch.nan_to_num(skin_smoothness_loss, nan=0.0)
        
        # Adjust component weights
        total_loss = (
            l1_loss * 0.2 +
            perceptual_loss * 0.15 +
            color_loss * 0.15 +
            skin_color_loss * 0.3 +
            skin_smoothness_loss * 0.2
        )
        
        # Финальная проверка на NaN в общей потере
        total_loss = torch.nan_to_num(total_loss, nan=0.1)  # Если всё-таки получили NaN, используем 0.1 как дефолтное значение
        
        return total_loss, {
            'l1': l1_loss,
            'perceptual': perceptual_loss,
            'color': color_loss,
            'skin_color': skin_color_loss,
            'skin_smoothness': skin_smoothness_loss
        }

def rgb_to_lab(rgb):
    """Конвертация из RGB в LAB цветовое пространство с защитой от NaN."""
    # Нормализация RGB в диапазон [0, 1]
    rgb = (rgb + 1) / 2
    
    # Добавим очистку NaN или бесконечностей, если они уже есть
    rgb = torch.nan_to_num(rgb, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Матрица преобразования RGB -> XYZ
    rgb_to_xyz = torch.tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ]).to(rgb.device)
    
    # RGB -> XYZ
    xyz = torch.matmul(rgb.permute(0, 2, 3, 1), rgb_to_xyz.t())
    
    # XYZ -> LAB (приближенное преобразование)
    # Добавляем маленькое значение, чтобы избежать деления на ноль
    epsilon = 1e-6
    reference_white = torch.tensor([0.950456, 1.0, 1.088754]).to(xyz.device)
    
    # Нормализация относительно белой точки и защита от деления на ноль
    xyz_normalized = xyz / (reference_white + epsilon)
    
    # Очистим от возможных NaN перед дальнейшей обработкой
    xyz_normalized = torch.nan_to_num(xyz_normalized, nan=0.01, posinf=1.0, neginf=0.01)
    
    # Применяем нелинейное преобразование (кубический корень для больших значений)
    mask = xyz_normalized > 0.008856
    xyz_normalized_cuberoot = torch.pow(torch.clamp(xyz_normalized, min=0.008856), 1/3)
    xyz_normalized_linear = 7.787 * xyz_normalized + 16/116
    
    # Применяем разные формулы в зависимости от значения
    xyz_normalized = torch.where(mask, xyz_normalized_cuberoot, xyz_normalized_linear)
    
    # Вычисляем LAB компоненты
    L = torch.clamp((116 * xyz_normalized[:, :, :, 1] - 16) / 100, 0.0, 1.0)
    a = torch.clamp(500 * (xyz_normalized[:, :, :, 0] - xyz_normalized[:, :, :, 1]) / 127, -1.0, 1.0)
    b = torch.clamp(200 * (xyz_normalized[:, :, :, 1] - xyz_normalized[:, :, :, 2]) / 127, -1.0, 1.0)
    
    # Собираем компоненты и еще раз проверяем на NaN
    lab = torch.stack([L, a, b], dim=1)
    lab = torch.nan_to_num(lab, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return lab

def setup_model(weights_path=None, device=None):
    """Настраивает модель для дообучения."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {device}")
    
    arch = models.resnet34
    model = UNetGeneratorModule(
        arch, 
        n_classes=3
    ).to(device)
    
    # Выводим размеры модели для проверки только в начале, без деталей
    dummy_input = torch.zeros(1, 3, 256, 256).to(device)
    try:
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Размер выхода модели: {output.shape}")
            assert output.shape[2:] == (256, 256), "Размер выхода модели не соответствует ожидаемому (256x256)"
    except Exception as e:
        print(f"Ошибка при проверке модели: {e}")
    
    if weights_path and os.path.exists(weights_path):
        print(f"Загрузка весов из {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            
            # Проверяем формат словаря весов
            if 'model_state_dict' in state_dict:
                # Проверяем на совместимость ключей
                model_dict = model.state_dict()
                state_dict_filtered = {k: v for k, v in state_dict['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
                print(f"Загружено {len(state_dict_filtered)}/{len(model_dict)} слоев из сохраненной модели")
                model.load_state_dict(state_dict_filtered, strict=False)
            else:
                # Проверяем на совместимость ключей
                model_dict = model.state_dict()
                state_dict_filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                print(f"Загружено {len(state_dict_filtered)}/{len(model_dict)} слоев из сохраненной модели")
                model.load_state_dict(state_dict_filtered, strict=False)
        except Exception as e:
            print(f"Ошибка при загрузке весов: {e}")
            
    print("Продолжение с инициализацией по умолчанию")
    
    return model

def finetune(data_dir, output_dir, weights_path=None, num_epochs=10, 
             batch_size=16, learning_rate=1e-5, img_size=256):
    """Дообучение модели с использованием предобработанных данных."""
    
    # Загружаем информацию о количестве обработанных изображений
    info_file = os.path.join(data_dir, 'processed_info.npy')
    if os.path.exists(info_file):
        processed_info = np.load(info_file, allow_pickle=True).item()
        print(f"Найдено обработанных изображений:")
        print(f"- Всего: {processed_info['total_processed']}")
        print(f"- Тренировочных: {processed_info['training_size']}")
        print(f"- Валидационных: {processed_info['validation_size']}")
    else:
        raise Exception("Не найдена информация об обработанных изображениях!")
    
    # Проверяем, достаточно ли данных для обучения
    min_required = batch_size * 2  # Минимум 2 батча
    if processed_info['training_size'] < min_required:
        raise Exception(
            f"Недостаточно данных для обучения! "
            f"Необходимо минимум {min_required} изображений, "
            f"доступно {processed_info['training_size']}"
        )
    
    # Получаем загрузчики данных с актуальным количеством изображений
    train_loader, val_loader = get_dataloaders(
        data_dir, 
        batch_size=min(batch_size, processed_info['training_size']), 
        img_size=img_size
    )
    
    print(f"Подготовка к обучению:")
    print(f"- Размер батча: {batch_size}")
    print(f"- Количество эпох: {num_epochs}")
    print(f"- Learning rate: {learning_rate}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # 1. Добавляем VGG для перцептивной loss
    vgg = models.vgg16(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    # Настройка модели и оптимизатора
    model = setup_model(weights_path, device)
    criterion = CombinedLoss(vgg).to(device)
    
    # Downgraded optimizer settings to be less memory-intensive
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-3,  # Lower learning rate
        weight_decay=0.01,  # Less aggressive weight decay
        betas=(0.9, 0.999)  # Default betas
    )
    
    # Simpler learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-3,  # Lower max learning rate
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    
    # Отслеживание метрик
    best_val_loss = float('inf')
    train_metrics = {'total': [], 'l1': [], 'perceptual': [], 'color': [], 'skin_color': [], 'skin_smoothness': []}
    val_metrics = {'total': [], 'l1': [], 'perceptual': [], 'color': [], 'skin_color': [], 'skin_smoothness': []}
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        epoch_metrics = {'total': 0, 'l1': 0, 'perceptual': 0, 'color': 0, 'skin_color': 0, 'skin_smoothness': 0}
        
        for batch in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs}"):
            bw_images = batch['bw'].to(device)
            color_images = batch['color'].to(device)
            
            # Simple forward pass without mixup
            optimizer.zero_grad()
            outputs = model(bw_images)
            loss, components = criterion(outputs, color_images)
            
            loss.backward()
            
            # Standard gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Обновление метрик
            epoch_metrics['total'] += loss.item()
            for k, v in components.items():
                if isinstance(v, torch.Tensor):
                    epoch_metrics[k] += v.item()
                else:
                    epoch_metrics[k] += v
        
        scheduler.step()
        
        # Нормализация метрик
        for k in epoch_metrics:
            epoch_metrics[k] /= len(train_loader)
            train_metrics[k].append(epoch_metrics[k])
        
        # Валидация
        model.eval()
        val_epoch_metrics = {'total': 0, 'l1': 0, 'perceptual': 0, 'color': 0, 'skin_color': 0, 'skin_smoothness': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                bw_images = batch['bw'].to(device)
                color_images = batch['color'].to(device)
                outputs = model(bw_images)
                loss, components = criterion(outputs, color_images)
                
                val_epoch_metrics['total'] += loss.item()
                for k, v in components.items():
                    if isinstance(v, torch.Tensor):
                        val_epoch_metrics[k] += v.item()
                    else:
                        val_epoch_metrics[k] += v
        
        # Нормализация валидационных метрик
        for k in val_epoch_metrics:
            val_epoch_metrics[k] /= len(val_loader)
            val_metrics[k].append(val_epoch_metrics[k])
        
        # Сохранение лучшей модели
        if val_epoch_metrics['total'] < best_val_loss:
            best_val_loss = val_epoch_metrics['total']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, f"{output_dir}/best_model.pth")
        
        # Вывод метрик
        print(f"\nЭпоха {epoch+1} метрики:")
        print("Обучение:", {k: f"{v:.4f}" for k, v in epoch_metrics.items()})
        print("Валидация:", {k: f"{v:.4f}" for k, v in val_epoch_metrics.items()})
    
    return model, train_metrics, val_metrics

def visualize_results(model, data_loader, output_dir, num_samples=5):
    """Визуализирует результаты модели."""
    device = next(model.parameters()).device
    model.eval()
    
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
                
            bw_images = batch['bw'].to(device)
            color_images = batch['color'].to(device)
            
            outputs = model(bw_images)
            
            # Если размеры не совпадают, интерполируем
            if outputs.shape != color_images.shape:
                outputs = nn.functional.interpolate(
                    outputs, 
                    size=color_images.shape[2:],
                    mode='bilinear', 
                    align_corners=True
                )
            
            # Визуализация
            for j in range(min(3, len(bw_images))):  # Показываем максимум 3 изображения из батча
                # Преобразование тензоров в изображения
                bw_img = ((bw_images[j].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)

def get_free_space(path):
    """Возвращает количество свободного места на диске в байтах"""
    import shutil
    # Если путь не существует, используем родительскую директорию
    if not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free

def estimate_required_space(num_samples, img_size):
    """
    Оценивает необходимое место на диске
    Возвращает размер в байтах
    """
    # Размер одного изображения (RGB float32) = высота * ширина * 3 канала * 4 байта
    single_img_size = img_size * img_size * 3 * 4
    # Для каждого изображения создаются две версии (ч/б и цветная)
    total_size = single_img_size * 2 * num_samples
    # Добавляем 5% запаса вместо 10%
    return int(total_size * 1.05)

def process_and_prepare_data(source_dir=FFHQ_PATH, output_dir=PROCESSED_DATA_DIR, num_samples=52000, img_size=256):
    """
    Обрабатывает датасет FFHQ и создает структуру данных для обучения
    """
    # Создаем базовую директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Проверяем доступное место
    free_space = get_free_space(output_dir)
    space_per_image = estimate_required_space(1, img_size)  # Размер для одного изображения
    
    # Оставляем 1GB для системных нужд
    usable_space = free_space - (1 * 1024**3)
    
    # Рассчитываем, сколько изображений можем обработать
    # Делим на 2.1 вместо 2 для учета небольшого оверхеда файловой системы
    possible_images = int(usable_space / (space_per_image / 2.1))
    actual_samples = min(num_samples, possible_images)
    
    print(f"Анализ доступного места:")
    print(f"- Всего доступно: {free_space / (1024**3):.1f} GB")
    print(f"- Используемое место: {usable_space / (1024**3):.1f} GB")
    print(f"- Размер одного изображения: {space_per_image / (1024**2):.1f} MB")
    print(f"- Можно обработать максимум: {possible_images} изображений")
    print(f"- Будет обработано: {actual_samples} изображений")
    print(f"- Ожидаемое использование места: {(actual_samples * space_per_image / 2.1) / (1024**3):.1f} GB")
    
    # Создаем структуру директорий
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    for dir_path in [
        train_dir + "/bw", train_dir + "/color",
        val_dir + "/bw", val_dir + "/color"
    ]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Получаем список файлов
    image_files = list(Path(source_dir).rglob('*.png')) + \
                 list(Path(source_dir).rglob('*.jpg')) + \
                 list(Path(source_dir).rglob('*.jpeg'))
    
    # Ограничиваем количество файлов доступным местом
    image_files = image_files[:actual_samples]
    total_files = len(image_files)
    
    print(f"Найдено изображений: {total_files}")
    
    val_size = int(total_files * 0.1)
    processed = 0
    errors = 0
    
    pbar = tqdm(
        total=total_files,
        desc="Обработка изображений",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    
    for idx, img_path in enumerate(image_files):
        try:
            # Проверяем, осталось ли место для текущего изображения
            if get_free_space(output_dir) < space_per_image * 2:
                print(f"\n⚠️ Место на диске закончилось. Останавливаем обработку.")
                print(f"Успешно обработано {processed} изображений.")
                break
            
            # Обработка изображения
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            
            bw_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            bw_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2RGB)
            
            img = img.astype(np.float32) / 255.0
            bw_img = bw_img.astype(np.float32) / 255.0
            
            is_val = idx < val_size
            save_dir = val_dir if is_val else train_dir
            
            np.save(os.path.join(save_dir, "color", f"img_{idx:05d}.npy"), img)
            np.save(os.path.join(save_dir, "bw", f"img_{idx:05d}.npy"), bw_img)
            
            processed += 1
            
            pbar.set_postfix({
                'Прогресс': f'{processed/total_files*100:.1f}%',
                'Ошибки': errors,
                'Успешно': processed,
                'Свободно': f'{get_free_space(output_dir)/(1024**3):.1f}GB'
            })
            pbar.update(1)
            
        except Exception as e:
            errors += 1
            print(f"\nОшибка при обработке {img_path}: {e}")
            continue
    
    pbar.close()
    
    # Сохраняем информацию о количестве обработанных изображений
    processed_info = {
        'total_processed': processed,
        'validation_size': min(val_size, processed),
        'training_size': processed - min(val_size, processed)
    }
    
    # Сохраняем информацию в файл
    info_file = os.path.join(output_dir, 'processed_info.npy')
    np.save(info_file, processed_info)
    
    print(f"\nОбработка завершена!")
    print(f"- Всего планировалось: {total_files}")
    print(f"- Успешно обработано: {processed} ({processed/total_files*100:.1f}%)")
    print(f"- Ошибок обработки: {errors} ({errors/total_files*100:.1f}%)")
    print(f"- Тренировочных: {processed_info['training_size']}")
    print(f"- Валидационных: {processed_info['validation_size']}")
    print(f"- Оставшееся место: {get_free_space(output_dir)/(1024**3):.1f} GB")
    print(f"- Данные сохранены в: {output_dir}")
    
    return output_dir, processed_info

def visualize_skin_mask(model, data_loader, output_dir, num_samples=10):
    """Visualizes skin masks using the approach from mask.py"""
    device = next(model.parameters()).device
    vgg = models.vgg16(pretrained=True).features.to(device).eval()
    criterion = CombinedLoss(vgg)
    
    os.makedirs(os.path.join(output_dir, 'skin_masks'), exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
            
            color_images = batch['color'].to(device)
            
            # Get skin mask using mask.py approach
            skin_masks = criterion.get_improved_skin_mask(color_images)
            
            # Save visualizations
            for j in range(len(color_images)):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                img = (color_images[j].cpu().numpy().transpose(1, 2, 0) + 1) / 2
                axes[0].imshow(img)
                axes[0].set_title('Original')
                
                # Final skin mask
                mask = skin_masks[j, 0].cpu().numpy()
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Skin Mask')
                
                # Masked image
                masked_img = img.copy()
                for c in range(3):
                    masked_img[:, :, c] = masked_img[:, :, c] * mask
                axes[2].imshow(masked_img)
                axes[2].set_title('Masked Image')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'skin_masks', f'skin_mask_{i}_{j}.png'))
                plt.close()

def enhanced_skin_loss(output, target, skin_mask):
    """Enhanced loss for facial skin areas using mask.py approach"""
    # Convert to LAB
    output_lab = rgb_to_lab(output)
    target_lab = rgb_to_lab(target)
    
    # L-channel (brightness) loss with lower weight
    l_loss = F.l1_loss(output_lab[:, 0:1] * skin_mask, 
                       target_lab[:, 0:1] * skin_mask)
    
    # A and B channels (color) loss with higher weight
    ab_loss = F.smooth_l1_loss(output_lab[:, 1:] * skin_mask, 
                              target_lab[:, 1:] * skin_mask)
    
    # Smoothness loss
    grad_x = torch.abs(output_lab[:, 1:, :, :-1] - output_lab[:, 1:, :, 1:])
    grad_y = torch.abs(output_lab[:, 1:, :-1, :] - output_lab[:, 1:, 1:, :])
    smoothness_loss = torch.mean(grad_x * skin_mask[:, :, :, :-1] + 
                                grad_y * skin_mask[:, :, :-1, :])
    
    return l_loss * 0.5 + ab_loss * 2.0 + smoothness_loss * 1.5

class SkinAwareColorization(nn.Module):
    """Специальный слой, обрабатывающий кожу отдельно от остального изображения"""
    def __init__(self, n_features):
        super().__init__()
        self.skin_features = nn.Sequential(
            nn.Conv2d(n_features, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1)
        )
        
        self.global_features = nn.Sequential(
            nn.Conv2d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1)
        )
        
    def forward(self, features, skin_mask):
        # Отдельно обрабатываем особенности кожи
        skin_colors = self.skin_features(features) 
        # Обрабатываем глобальные особенности
        global_colors = self.global_features(features)
        
        # Комбинируем результаты с использованием маски
        combined = skin_colors * skin_mask + global_colors * (1 - skin_mask)
        return combined

def skin_aware_augmentation(image, skin_mask):
    """Аугментация с акцентом на кожу"""
    # Конвертируем в LAB
    lab_img = rgb_to_lab_np(image)
    
    # Случайно изменяем оттенок и насыщенность кожи для обогащения данных
    if np.random.random() > 0.5:
        # Меняем только a и b каналы кожи
        skin_pixels = skin_mask > 0.5
        if np.any(skin_pixels):
            # Случайное смещение оттенка (a канал)
            a_shift = np.random.uniform(-10, 10)
            lab_img[..., 1][skin_pixels] += a_shift
            
            # Случайное смещение насыщенности (b канал)
            b_shift = np.random.uniform(-10, 10)
            lab_img[..., 2][skin_pixels] += b_shift
    
    # Обратно в RGB
    augmented_img = lab_to_rgb_np(lab_img)
    return augmented_img

if __name__ == "__main__":
    print("Проверка окружения:")
    print(f"PyTorch version: {torch.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Проверяем наличие обработанных данных
    processed_info_file = os.path.join(PROCESSED_DATA_DIR, 'processed_info.npy')
    if os.path.exists(processed_info_file):
        print("\nНайдены обработанные данные, пропускаем этап обработки...")
        processed_info = np.load(processed_info_file, allow_pickle=True).item()
        processed_data_dir = PROCESSED_DATA_DIR
    else:
        print("\nОбработанные данные не найдены, начинаем обработку...")
        # Обработка исходных данных с заданным количеством изображений
        processed_data_dir, processed_info = process_and_prepare_data(
            source_dir=FFHQ_PATH,
            output_dir=PROCESSED_DATA_DIR,
            num_samples=12000,
            img_size=256
        )
    
    # Проверяем, достаточно ли данных для обучения
    if processed_info['total_processed'] > 0:
        print("\nНачинаем обучение на доступных данных...")
    
    # Создание директории для выходных данных
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Дообучение модели
    model, train_metrics, val_metrics = finetune(
            data_dir=processed_data_dir,
        output_dir=OUTPUT_DIR,
            weights_path=None,
            num_epochs=30,  
            batch_size=32,
            learning_rate=1e-3,
        img_size=256
    )
    # Визуализация результатов
    if processed_info['validation_size'] > 0:
            _, val_loader = get_dataloaders(processed_data_dir, batch_size=5, img_size=256)
    visualize_results(model, val_loader, OUTPUT_DIR)
    
        # Визуализация масок кожи
    if processed_info['validation_size'] > 0:
        _, val_loader = get_dataloaders(processed_data_dir, batch_size=5, img_size=256)
        visualize_skin_mask(model, val_loader, OUTPUT_DIR)
        print("Обучение и визуализация завершены!")
    else:
        print("\nНедостаточно данных для начала обучения!")