import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def compute_histogram(image):
    histogram = np.zeros(256, dtype=np.int32)
    height, width = image.shape
    
    for i in range(height):
        for j in range(width):
            pixel_value = image[i, j]
            histogram[pixel_value] += 1
    
    return histogram

def compute_cdf(histogram):
    cdf = np.zeros_like(histogram, dtype=np.float32)
    cdf[0] = histogram[0]
    
    for i in range(1, len(histogram)):
        cdf[i] = cdf[i-1] + histogram[i]
    
    return cdf

def normalize_cdf(cdf, total_pixels):
    cdf_min = np.min(cdf[cdf > 0])  # Минимальное ненулевое значение CDF
    cdf_normalized = np.zeros_like(cdf, dtype=np.uint8)
    
    for i in range(len(cdf)):
        # Формула выравнивания: round((cdf[i] - cdf_min) * 255 / (total_pixels - cdf_min))
        if cdf[i] > 0:
            cdf_normalized[i] = round((cdf[i] - cdf_min) * 255 / (total_pixels - cdf_min))
        else:
            cdf_normalized[i] = 0
    
    return cdf_normalized

def histogram_equalization_grayscale(image):
    height, width = image.shape
    total_pixels = height * width
    
    hist = compute_histogram(image)
    
    # Шаг 2: Вычисляем CDF
    cdf = compute_cdf(hist)
    
    # Шаг 3: Нормализуем CDF для получения lookup table
    lookup_table = normalize_cdf(cdf, total_pixels)
    
    # Шаг 4: Применяем преобразование к каждому пикселю
    equalized = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            original_value = image[i, j]
            equalized[i, j] = lookup_table[original_value]
    
    return equalized, hist, lookup_table

def rgb_to_grayscale_manual(image):
    height, width, channels = image.shape
    gray = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            # Усредняем значения трех каналов
            gray[i, j] = int((int(image[i, j, 0]) + int(image[i, j, 1]) + int(image[i, j, 2])) / 3)
    
    return gray

def histogram_equalization_rgb(image, mode='luminance'):
   
    height, width, channels = image.shape
    
    if mode == 'luminance':
        # Режим 1: Grayscale -> Equalization -> RGB
        gray = rgb_to_grayscale_manual(image)
        equalized_gray, hist, _ = histogram_equalization_grayscale(gray)
        
        # Преобразуем обратно в 3-канальное для отображения
        equalized_rgb = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)
        
        return equalized_rgb, hist, gray, equalized_gray
    
    elif mode == 'channel':
        # Режим 2: Выравнивание каждого канала отдельно
        equalized_rgb = np.zeros_like(image)
        channels_hist = []
        
        for c in range(channels):
            channel = image[:, :, c]
            equalized_channel, hist, _ = histogram_equalization_grayscale(channel)
            equalized_rgb[:, :, c] = equalized_channel
            channels_hist.append(hist)
        
        return equalized_rgb, channels_hist, None, None
    
    elif mode == 'value':
        # Режим 3: HSV - выравнивание только канала яркости
        # Вручную конвертируем BGR в HSV (упрощенно)
        hsv = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                b, g, r = image[i, j, 0] / 255.0, image[i, j, 1] / 255.0, image[i, j, 2] / 255.0
                
                # Value (яркость) - максимум из RGB
                v = max(r, g, b)
                
                # Упрощенное преобразование (для демонстрации)
                hsv[i, j, 2] = int(v * 255)  # V канал
        
        # Выравниваем только V канал
        v_channel = hsv[:, :, 2]
        equalized_v, hist, _ = histogram_equalization_grayscale(v_channel)
        
        # Собираем обратно (упрощенно)
        result = image.copy()
        for i in range(height):
            for j in range(width):
                # Масштабируем RGB пропорционально изменению V
                if v_channel[i, j] > 0:
                    scale = equalized_v[i, j] / v_channel[i, j]
                    result[i, j, 0] = np.clip(int(image[i, j, 0] * scale), 0, 255)
                    result[i, j, 1] = np.clip(int(image[i, j, 1] * scale), 0, 255)
                    result[i, j, 2] = np.clip(int(image[i, j, 2] * scale), 0, 255)
        
        return result, hist, v_channel, equalized_v

def plot_histograms(original_hist, equalized_hist, save_path=None):
    """
    Визуализация гистограмм до и после выравнивания
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(256), original_hist, color='blue', alpha=0.7, width=1)
    plt.title('Гистограмма ДО выравнивания')
    plt.xlabel('Яркость')
    plt.ylabel('Количество пикселей')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(256), equalized_hist, color='green', alpha=0.7, width=1)
    plt.title('Гистограмма ПОСЛЕ выравнивания')
    plt.xlabel('Яркость')
    plt.ylabel('Количество пикселей')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def apply_histogram_equalization_to_my_image():
    """
    Применение выравнивания гистограммы к вашему изображению
    """
    print("=" * 70)
    print("ВЫРАВНИВАНИЕ ГИСТОГРАММЫ - ВАШЕ ИЗОБРАЖЕНИЕ")
    print("=" * 70)
    
    # ============================================
    # ПУТЬ К ВАШЕМУ ИЗОБРАЖЕНИЮ
    # ============================================
    image_path = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\input_image.jpg'
    
    # Загружаем изображение
    print(f"\n📂 Загружаем изображение...")
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print("❌ Ошибка: Не удалось загрузить изображение!")
        return
    
    print(f"✅ Изображение загружено!")
    print(f"   Размер: {original_image.shape[1]}x{original_image.shape[0]}")
    
    # Создаем папку для результатов
    results_folder = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\histogram_results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"📁 Создана папка: {results_folder}")
    
    # Сохраняем оригинал
    cv2.imwrite(os.path.join(results_folder, '1_original.jpg'), original_image)
    
    # Преобразуем в оттенки серого
    gray_image = rgb_to_grayscale_manual(original_image)
    cv2.imwrite(os.path.join(results_folder, '2_grayscale.jpg'), gray_image)
    
    # ============================================
    # 1. ВЫРАВНИВАНИЕ ДЛЯ GRAYSCALE
    # ============================================
    print("\n🔄 1. Выравнивание гистограммы (grayscale)...")
    
    equalized_gray, hist_gray, lookup_table = histogram_equalization_grayscale(gray_image)
    cv2.imwrite(os.path.join(results_folder, '3_equalized_grayscale.jpg'), equalized_gray)
    
    # Вычисляем гистограмму после выравнивания
    hist_equalized = compute_histogram(equalized_gray)
    
    # Визуализируем гистограммы
    plot_histograms(hist_gray, hist_equalized, 
                   os.path.join(results_folder, '4_histogram_comparison.png'))
    
    print("   ✅ Grayscale - готово")
    
    # ============================================
    # 2. ВЫРАВНИВАНИЕ ДЛЯ RGB (РАЗНЫЕ РЕЖИМЫ)
    # ============================================
    print("\n🔄 2. Выравнивание гистограммы (RGB)...")
    
    # Режим luminance
    print("   ⏳ Режим 'luminance'...")
    equalized_luminance, hist_lum, _, _ = histogram_equalization_rgb(original_image, mode='luminance')
    cv2.imwrite(os.path.join(results_folder, '5_equalized_luminance.jpg'), equalized_luminance)
    print("   ✅ luminance - готово")
    
    # Режим channel
    print("   ⏳ Режим 'channel'...")
    equalized_channel, hists_channel, _, _ = histogram_equalization_rgb(original_image, mode='channel')
    cv2.imwrite(os.path.join(results_folder, '6_equalized_channel.jpg'), equalized_channel)
    print("   ✅ channel - готово")
    
    # Режим value (упрощенный HSV)
    print("   ⏳ Режим 'value'...")
    equalized_value, hist_value, _, _ = histogram_equalization_rgb(original_image, mode='value')
    cv2.imwrite(os.path.join(results_folder, '7_equalized_value.jpg'), equalized_value)
    print("   ✅ value - готово")
    
    # ============================================
    # 3. СОЗДАЕМ СРАВНИТЕЛЬНЫЕ ИЗОБРАЖЕНИЯ
    # ============================================
    print("\n🎨 3. Создаем сравнительные изображения...")
    
    # Сравнение grayscale
    compare_gray = np.hstack([
        cv2.resize(gray_image, (400, 300)),
        cv2.resize(equalized_gray, (400, 300))
    ])
    cv2.putText(compare_gray, 'Grayscale оригинал', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.putText(compare_gray, 'После выравнивания', (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.imwrite(os.path.join(results_folder, '8_compare_grayscale.jpg'), compare_gray)
    
    # Сравнение всех режимов RGB
    compare_rgb = np.hstack([
        cv2.resize(original_image, (300, 200)),
        cv2.resize(equalized_luminance, (300, 200)),
        cv2.resize(equalized_channel, (300, 200)),
        cv2.resize(equalized_value, (300, 200))
    ])
    cv2.putText(compare_rgb, 'Оригинал', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(compare_rgb, 'Luminance', (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(compare_rgb, 'Channel', (610, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(compare_rgb, 'Value', (910, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(results_folder, '9_compare_rgb_modes.jpg'), compare_rgb)
    
    # ============================================
    # 4. ПОКАЗЫВАЕМ РЕЗУЛЬТАТЫ
    # ============================================
    print("\n🖼️ 4. Показываем результаты...")
    
    # Создаем окна
    windows = [
        ('Оригинал (цветной)', original_image),
        ('Grayscale', gray_image),
        ('Выравнивание Grayscale', equalized_gray),
        ('RGB - Luminance', equalized_luminance),
        ('RGB - Channel', equalized_channel),
        ('RGB - Value', equalized_value)
    ]
    
    for name, img in windows:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 600, 400)
        cv2.imshow(name, img)
    
    # Показываем гистограммы
    print("\n📊 Показываем гистограммы...")
    plot_histograms(hist_gray, hist_equalized)
    
    # ============================================
    # 5. ИТОГОВЫЙ ОТЧЕТ
    # ============================================
    print("\n" + "=" * 70)
    print("✅ ВЫРАВНИВАНИЕ ГИСТОГРАММЫ ВЫПОЛНЕНО!")
    print("=" * 70)
    
    print(f"\n📁 Все результаты сохранены в папке:")
    print(f"   {results_folder}")
    
    print("\n📋 Созданные файлы:")
    files = os.listdir(results_folder)
    for i, file in enumerate(sorted(files), 1):
        print(f"   {i:2d}. {file}")
    
    print(f"\n📊 Всего создано файлов: {len(files)}")
    print(f"\n⌨️ Нажмите любую клавишу в окне изображения для закрытия...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Открываем папку с результатами
    os.startfile(results_folder)
    
    return results_folder

def analyze_histogram_stats(image):
    """
    Анализ статистик гистограммы
    """
    hist = compute_histogram(image)
    
    # Основные статистики
    mean = np.mean(image)
    std = np.std(image)
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Находим наиболее частые значения
    most_frequent = np.argmax(hist)
    
    print(f"\n📊 Статистика изображения:")
    print(f"   Среднее: {mean:.2f}")
    print(f"   Стандартное отклонение: {std:.2f}")
    print(f"   Минимум: {min_val}")
    print(f"   Максимум: {max_val}")
    print(f"   Наиболее частый уровень: {most_frequent} (встречается {hist[most_frequent]} раз)")
    
    return hist

def quick_histogram_test():
    """
    Быстрый тест выравнивания гистограммы
    """
    print("\n" + "=" * 70)
    print("БЫСТРЫЙ ТЕСТ ВЫРАВНИВАНИЯ ГИСТОГРАММЫ")
    print("=" * 70)
    
    image_path = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\input_image.jpg'
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Изображение не найдено!")
        return
    
    gray = rgb_to_grayscale_manual(img)
    
    print("\n🔬 Анализ исходного изображения:")
    analyze_histogram_stats(gray)
    
    print("\n🔄 Применяем выравнивание...")
    equalized, hist_orig, _ = histogram_equalization_grayscale(gray)
    
    print("\n📊 Анализ после выравнивания:")
    analyze_histogram_stats(equalized)
    
    # Показываем результаты
    cv2.imshow('Оригинал (цветной)', img)
    cv2.imshow('Grayscale оригинал', gray)
    cv2.imshow('После выравнивания', equalized)
    
    # Показываем гистограммы
    plot_histograms(hist_orig, compute_histogram(equalized))
    
    print("\n⌨️ Нажмите любую клавишу для закрытия...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ОСНОВНОЙ ЗАПУСК
    apply_histogram_equalization_to_my_image()
    
    # РАСКОММЕНТИРУЙТЕ ДЛЯ БЫСТРОГО ТЕСТА:
    # quick_histogram_test()
