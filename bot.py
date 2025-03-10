import os
import logging
import numpy as np
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import ContentType
from aiogram.utils.executor import start_polling
from PIL import Image
import torch
from deoldify.visualize import get_image_colorizer


load_dotenv(dotenv_path=".env", override=True)

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Error: BOT_TOKEN isn't found!")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

colorizer_artistic = get_image_colorizer(artistic=True)
colorizer_stable = get_image_colorizer(artistic=False)


# Функция цветизации (с выбором модели)
def process_colorization(image_path, artistic=True):
    colorizer = colorizer_artistic if artistic else colorizer_stable
    colorized_image = colorizer.get_transformed_image(image_path, render_factor=35)
    return colorized_image


# Функция обработки команды /start
@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.reply("Привет! Отправь мне изображение, и я улучшу его качество или добавлю цвета! 🎨\n"
                        "Напиши /mode_artistic или /mode_stable, чтобы выбрать стиль цветизации.")


# Обработка выбора модели цветизации
colorization_mode = {"artistic": True}


@dp.message_handler(commands=['mode_artistic'])
async def set_mode_artistic(message: types.Message):
    colorization_mode["artistic"] = True
    await message.reply("🎨 Режим цветизации: Artistic")


@dp.message_handler(commands=['mode_stable'])
async def set_mode_stable(message: types.Message):
    colorization_mode["artistic"] = False
    await message.reply("🎨 Режим цветизации: Stable")


# Функция обработки изображений
@dp.message_handler(content_types=[ContentType.PHOTO])
async def handle_photo(message: types.Message):
    print("📸 Получено фото!")  # Отладочный вывод

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_path = file.file_path
    input_path = "input.jpg"
    output_path = "output.jpg"

    await bot.download_file(file_path, input_path)
    print("Фото загружено!")  # Отладочный вывод

    # Улучшенное определение ч/б фото
    image = Image.open(input_path).convert("RGB")
    img_array = np.array(image)
    
    # Проверяем разницу между RGB каналами
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    rg_diff = np.abs(r - g).mean()
    rb_diff = np.abs(r - b).mean()
    gb_diff = np.abs(g - b).mean()
    
    # Если разница между каналами минимальна, считаем изображение ч/б
    threshold = 5.0  # Можно настроить этот порог
    is_grayscale = (rg_diff < threshold and rb_diff < threshold and gb_diff < threshold)

    if is_grayscale:
        print("Цветизация фото...")
        mode = "artistic" if colorization_mode["artistic"] else "stable"
        result_image = process_colorization(input_path, artistic=colorization_mode["artistic"])
    else:
        print("Фото не распознано как черно-белое")
        await message.reply("❌ Ошибка: фото не распознано как черно-белое.")

    # Сохраняем результат
    result_image.save(output_path)
    print("Обработка завершена! Отправляю фото...")

    # Проверяем, существует ли обработанное изображение перед отправкой
    if os.path.exists(output_path):
        with open(output_path, "rb") as photo_file:
            await message.answer_photo(photo=photo_file)
    else:
        print("❌ Ошибка: обработанное изображение не найдено!")
        await message.reply("❌ Ошибка: не удалось обработать изображение.")

# Запуск бота
if __name__ == "__main__":
    start_polling(dp, skip_updates=True)
