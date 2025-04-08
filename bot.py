"""
Telegram бот для обработки изображений с функциями колоризации и улучшения качества.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict
from pathlib import Path

import aiohttp
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.types import ContentType
from aiogram.utils.executor import start_polling
from deoldify.visualize import get_image_colorizer
from deoldify.generators import gen_inference_wide
from deoldify.filters import MasterFilter, ColorizerFilter
from deoldify.visualize import ModelImageVisualizer
from dotenv import load_dotenv
from PIL import Image
import torch
import urllib3

INPUT_PATH: str = "input.jpg"
OUTPUT_PATH: str = "output.jpg"
GRAYSCALE_THRESHOLD: float = 5.0
PICWISH_API_URL: str = "https://techhk.aoscdn.com/api/tasks/visual/scale"

MODES: Dict[str, str] = {
    "colorize_artistic": "Художественная колоризация",
    "colorize_stable": "Стабильная колоризация",
    "enhance": "Улучшение качества",
    "compare_models": "Сравнение моделей"
}

BUTTONS: Dict[str, str] = {
    "🎨 Художественная колоризация": "colorize_artistic",
    "🖼 Стабильная колоризация": "colorize_stable",
    "✨ Улучшение качества": "enhance",
    "🔍 Сравнение моделей": "compare_models"
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Определение устройства для работы
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Используется устройство: {device}")

class ImageProcessor:
    """Класс для обработки изображений."""

    def __init__(self) -> None:
        """Инициализация обработчика изображений."""
        self.colorizer_artistic = get_image_colorizer(artistic=True)
        self.colorizer_stable = get_image_colorizer(artistic=False)
        # Создаем копию стабильного колоризатора для настроенной модели
        try:
            # Создаем стандартный стабильный колоризатор
            self.colorizer_tuned = get_image_colorizer(artistic=False)

            # 2. Загружаем наши весса
            model_path = 'models/Tuned_model.pth'
            if os.path.exists(model_path):
                logger.info(f"Загрузка весов модели из {model_path}")
                checkpoint = torch.load(model_path, map_location=device)

                # 3. Заменяем веса модели в существующем колоризаторе
                if 'model_state_dict' in checkpoint:
                    # Правильный путь к модели на основе анализа visualize.py
                    model = self.colorizer_tuned.filter.filters[0].learn.model

                    # Загружаем веса
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    model.eval()

                    logger.info("Настроенная модель успешно загружена")
                else:
                    logger.error("В файле модели не найден model_state_dict")
                    self.colorizer_tuned = None
            else:
                logger.error(f"Файл модели не найден: {model_path}")
                self.colorizer_tuned = None
        except Exception as e:
            logger.error(f"Подробная ошибка загрузки: {str(e)}", exc_info=True)
            self.colorizer_tuned = None

    def process_colorization(self, image_path: str, artistic: bool = True, model_type: str = 'stable') -> Image.Image:
        """
        Колоризация черно-белого изображения.

        Args:
            image_path: Путь к изображению
            artistic: Использовать художественную колоризацию
            model_type: Тип модели ('stable' или 'tuned')

        Returns:
            Обработанное изображение
        """
        try:
            logger.info(f"Запуск колоризации с параметрами artistic={artistic}, model_type={model_type}")
            if artistic:
                colorizer = self.colorizer_artistic
            else:
                if model_type == 'tuned' and self.colorizer_tuned is None:
                    logger.warning("Настроенная модель недоступна, использую стандартную модель")
                    colorizer = self.colorizer_stable
                else:
                    colorizer = self.colorizer_tuned if model_type == 'tuned' else self.colorizer_stable

            # Добавьте эти строки для отладки
            if not artistic:
                model_id = id(colorizer.filter.filters[0].learn.model)
                logger.info(f"ID модели {model_type}: {model_id}")

                # Для более детального сравнения
                if model_type == 'tuned':
                    tuned_id = id(self.colorizer_tuned.filter.filters[0].learn.model)
                    stable_id = id(self.colorizer_stable.filter.filters[0].learn.model)
                    logger.info(f"ID моделей для сравнения: tuned={tuned_id}, stable={stable_id}")

            logger.info(f"Выбран колоризатор: {model_type}")
            return colorizer.get_transformed_image(str(image_path), render_factor=35)
        except Exception as e:
            logger.error(f"Ошибка при колоризации: {str(e)}")
            raise

    async def enhance_image_picwish(self, image_path: str) -> bool:
        """
        Улучшение качества изображения через PicWish API.

        Args:
            image_path: Путь к изображению

        Returns:
            True если улучшение успешно, False в противном случае
        """
        try:
            headers = {"X-API-KEY": os.getenv("PICWISH_API_KEY")}
            data = {"sync": "1", "type": "face"}

            logger.info(f"Отправка запроса к PicWish API: {PICWISH_API_URL}")

            async with aiohttp.ClientSession() as session:
                form_data = aiohttp.FormData()
                form_data.add_field(
                    'image_file',
                    open(image_path, 'rb'),
                    filename='image.jpg',
                    content_type='image/jpeg'
                )

                for key, value in data.items():
                    form_data.add_field(key, value)

                async with session.post(PICWISH_API_URL, headers=headers, data=form_data) as response:
                    logger.info(f"Статус ответа: {response.status}")

                    if response.status == 200:
                        response_data = await response.json()
                        logger.info(f"Ответ API: {response_data}")

                        if "data" in response_data and "image" in response_data["data"]:
                            image_url = response_data["data"]["image"]
                            logger.info(f"URL обработанного изображения: {image_url}")

                            async with session.get(image_url) as img_response:
                                if img_response.status == 200:
                                    image_data = await img_response.read()
                                    with open(OUTPUT_PATH, "wb") as f:
                                        f.write(image_data)
                                    logger.info("Улучшенное изображение успешно сохранено")
                                    return True
                                logger.error(f"Ошибка при скачивании изображения: {img_response.status}")
                        else:
                            logger.error("Обработанное изображение не найдено в ответе API")
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка API PicWish: {response.status}")
                        logger.error(error_text)

                    return False

        except Exception as e:
            logger.error(f"Ошибка при улучшении изображения: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def check_if_grayscale(image_path: str) -> bool:
        """
        Проверка является ли изображение черно-белым.

        Args:
            image_path: Путь к изображению

        Returns:
            True если изображение черно-белое, False в противном случае
        """
        # Улучшенное определение ч/б фото
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)

        # Проверяем разницу между RGB каналами
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        rg_diff = np.abs(r - g).mean()
        rb_diff = np.abs(r - b).mean()
        gb_diff = np.abs(g - b).mean()

        # Если разница между каналами минимальна, считаем изображение ч/б
        threshold = 5.0  # Можно настроить этот порог
        is_grayscale = (rg_diff < threshold and rb_diff < threshold and gb_diff < threshold)

        logger.info(f"Проверка ч/б: rg_diff={rg_diff}, rb_diff={rb_diff}, gb_diff={gb_diff}, результат={is_grayscale}")
        return is_grayscale

    # Вспомогательная функция для исследования структуры объекта
    def inspect_object(self, obj, name="object", max_depth=2, current_depth=0):
        if current_depth > max_depth:
            return

        logger.info(f"{'  ' * current_depth}Исследуем {name} типа {type(obj).__name__}")

        for attr_name in dir(obj):
            if attr_name.startswith('__'):
                continue
            try:
                attr = getattr(obj, attr_name)
                if not callable(attr) and not attr_name.startswith('_'):
                    logger.info(f"{'  ' * current_depth}- {attr_name}: {type(attr).__name__}")
                    if hasattr(attr, '__dict__') and current_depth < max_depth:
                        self.inspect_object(attr, f"{name}.{attr_name}", max_depth, current_depth + 1)
            except Exception as e:
                logger.info(f"{'  ' * current_depth}- Ошибка при доступе к {attr_name}: {e}")

    # Используйте эту функцию для анализа colorizer_tuned
    def inspect_colorizer_tuned(self):
        self.inspect_object(self.colorizer_tuned, "colorizer_tuned")

    # Получаем архитектуру стандартной модели для сравнения
    def inspect_stable_model(self):
        if hasattr(self.colorizer_stable, 'filter') and hasattr(self.colorizer_stable.filter, 'learn'):
            stable_model = self.colorizer_stable.filter.learn.model
            logger.info(f"Архитектура стандартной модели: {type(stable_model).__name__}")
            logger.info(f"Слои стандартной модели: {list(stable_model.children())[:3]}...")

    def get_stable_image_colorizer(
        self,
        root_folder: Path = Path('./'),
        weights_name: str = 'ColorizeStable_gen',
        results_dir='result_images',
        render_factor: int = 35
    ) -> ModelImageVisualizer:
        learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
        filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
        vis = ModelImageVisualizer(filtr, results_dir=results_dir)
        return vis

class TelegramBot:
    """Класс для работы с Telegram ботом."""

    def __init__(self) -> None:
        """Инициализация бота."""
        load_dotenv(dotenv_path=".env", override=True)

        bot_token = os.getenv("BOT_TOKEN")
        if not bot_token:
            raise ValueError("Error: BOT_TOKEN isn't found!")

        self.image_processor = ImageProcessor()
        self.current_mode = {"mode": "colorize_artistic"}
        self.bot = Bot(token=bot_token)
        self.dp = Dispatcher(self.bot)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Настройка обработчиков сообщений."""
        self.dp.register_message_handler(self.start, commands=['start'])
        self.dp.register_message_handler(
            self.handle_mode_selection,
            lambda message: message.text in BUTTONS.keys()
        )
        self.dp.register_message_handler(
            self.handle_photo,
            content_types=[ContentType.PHOTO]
        )

    async def start(self, message: types.Message) -> None:
        """Обработчик команды /start."""
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
        buttons = [types.KeyboardButton(text) for text in BUTTONS.keys()]
        keyboard.add(*buttons)

        await message.reply(
            "Привет! Я могу помочь с обработкой ваших фотографий! 🎨\n"
            "Выберите режим обработки:\n"
            "1. Художественная колоризация - для творческой раскраски ч/б фото\n"
            "2. Стабильная колоризация - для реалистичной раскраски ч/б фото\n"
            "3. Улучшение качества - для улучшения любых фотографий\n"
            "4. Сравнение моделей - сравнение обычной DeOldify и дообученной версии\n"
            "\nПосле выбора режима просто отправьте мне фотографию!",
            reply_markup=keyboard
        )

    async def handle_mode_selection(self, message: types.Message) -> None:
        """Обработчик выбора режима работы."""
        selected_mode = BUTTONS[message.text]
        self.current_mode["mode"] = selected_mode
        logger.info(f"Режим изменен на: {selected_mode}")
        await message.reply(f"✅ Режим установлен: {MODES[selected_mode]}")

    async def handle_photo(self, message: types.Message) -> None:
        """Обработчик загруженных фотографий."""
        try:
            processing_msg = await message.reply("⏳ Обработка изображения началась...")

            photo = message.photo[-1]
            file = await self.bot.get_file(photo.file_id)
            await self.bot.download_file(file.file_path, INPUT_PATH)
            logger.info(f"Фото загружено: {INPUT_PATH}")

            mode = self.current_mode["mode"]
            logger.info(f"Выбранный режим: {mode}")

            if mode in ["colorize_artistic", "colorize_stable"]:
                await processing_msg.edit_text("🧐 Проверка, является ли изображение черно-белым...")

                if self.image_processor.check_if_grayscale(INPUT_PATH):
                    await processing_msg.edit_text(
                        "🎨 Выполняется колоризация изображения...\n"
                        "Это может занять некоторое время."
                    )

                    artistic = mode == "colorize_artistic"
                    result = self.image_processor.process_colorization(
                        INPUT_PATH,
                        artistic=artistic,
                        model_type='stable'
                    )
                    result.save(OUTPUT_PATH)
                else:
                    await processing_msg.edit_text("❌ Это изображение не является черно-белым")
                    return

            elif mode == "enhance":
                await processing_msg.edit_text(
                    "🔍 Улучшаем качество изображения...\n"
                    "Это может занять некоторое время."
                )

                logger.info("Начинаю улучшение качества изображения через PicWish API")
                success = await self.image_processor.enhance_image_picwish(INPUT_PATH)
                logger.info(f"Результат улучшения через API: {success}")

                if not success:
                    await processing_msg.edit_text("❌ Не удалось улучшить качество изображения.")
                    return

            elif mode == "compare_models":
                await processing_msg.edit_text("🧐 Проверка, является ли изображение черно-белым...")

                if self.image_processor.check_if_grayscale(INPUT_PATH):
                    await processing_msg.edit_text(
                        "🎨 Выполняется колоризация изображения двумя моделями...\n"
                        "Это может занять некоторое время."
                    )

                    # Проверяем доступность дообученной модели
                    if self.image_processor.colorizer_tuned is None:
                        await processing_msg.edit_text(
                            "❌ Дообученная модель недоступна. Пожалуйста, используйте другой режим."
                        )
                        return

                    # Обработка стандартной моделью
                    result_stable = self.image_processor.process_colorization(
                        INPUT_PATH,
                        artistic=False,
                        model_type='stable'
                    )
                    result_stable.save('output_stable.jpg')

                    # Обработка дообученной моделью
                    result_tuned = self.image_processor.process_colorization(
                        INPUT_PATH,
                        artistic=False,
                        model_type='tuned'
                    )
                    result_tuned.save('output_tuned.jpg')

                    # Отправка обоих результатов
                    media = types.MediaGroup()
                    media.attach_photo(types.InputFile('output_stable.jpg'), 'Стандартная модель DeOldify')
                    media.attach_photo(types.InputFile('output_tuned.jpg'), 'Дообученная модель')

                    await message.reply_media_group(media=media)
                    await processing_msg.delete()

                    # Очистка временных файлов
                    os.remove('output_stable.jpg')
                    os.remove('output_tuned.jpg')
                    return
                else:
                    await processing_msg.edit_text("❌ Это изображение не является черно-белым")
                    return

            if os.path.exists(OUTPUT_PATH):
                logger.info(f"Отправляю обработанное изображение {OUTPUT_PATH}")
                with open(OUTPUT_PATH, "rb") as photo_file:
                    await message.reply_photo(photo=photo_file, caption="✨ Готово!")
                await processing_msg.delete()
            else:
                logger.error(f"Файл результата не найден: {OUTPUT_PATH}")
                await processing_msg.edit_text(
                    "❌ Произошла ошибка при обработке изображения. "
                    "Файл результата не найден."
                )

        except Exception as e:
            logger.error(f"Ошибка при обработке фото: {str(e)}", exc_info=True)
            await message.reply(f"❌ Произошла ошибка: {str(e)}")

    async def start_polling(self) -> None:
        """Запуск бота."""
        try:
            await self.dp.start_polling()
        finally:
            await self.bot.session.close()
            await self.dp.storage.close()
            await self.dp.storage.wait_closed()

def main() -> None:
    """Точка входа в приложение."""
    try:
        logger.info("Бот запущен!")
        bot = TelegramBot()
        asyncio.run(bot.start_polling())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен!")

if __name__ == "__main__":
    main()