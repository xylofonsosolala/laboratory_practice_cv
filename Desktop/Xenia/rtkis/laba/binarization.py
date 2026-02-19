import cv2
import numpy as np
import os

def threshold_grayscale(image, threshold=127):
    """
    Ручная бинаризация для черно-белого изображения
    """
    height, width = image.shape
    binary = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            if image[i, j] >= threshold:
                binary[i, j] = 255
            else:
                binary[i, j] = 0
    
    return binary

def rgb_to_grayscale_manual(image):
    """
    Ручное преобразование RGB в оттенки серого (метод среднего)
    """
    height, width, channels = image.shape
    gray = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            # Усредняем значения трех каналов
            gray[i, j] = int((int(image[i, j, 0]) + int(image[i, j, 1]) + int(image[i, j, 2])) / 3)
    
    return gray

def threshold_rgb(image, threshold=127):
    """
    Бинаризация RGB изображения через преобразование в оттенки серого
    """
    # Сначала преобразуем RGB в оттенки серого вручную
    gray = rgb_to_grayscale_manual(image)
    
    # Затем бинаризуем полученное черно-белое изображение
    binary = threshold_grayscale(gray, threshold)
    
    # Преобразуем обратно в 3-канальное для отображения
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return binary_rgb, gray

def main():
    print("=" * 60)
    print("ПОРОГОВАЯ БИНАРИЗАЦИЯ ИЗОБРАЖЕНИЯ")
    print("=" * 60)
    
    # ============================================
    # ПУТЬ К ВАШЕМУ ИЗОБРАЖЕНИЮ
    # ============================================
    image_path = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\input_image.jpg'
    
    # Загружаем изображение
    print(f"\n📂 Загружаем изображение...")
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print("❌ Ошибка: Не удалось загрузить изображение!")
        print("   Проверьте путь:", image_path)
        return
    
    print(f"✅ Изображение загружено!")
    print(f"   Размер: {original_image.shape[1]}x{original_image.shape[0]}")
    print(f"   Каналы: {original_image.shape[2]}")
    
    # Создаем папку для результатов
    results_folder = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\threshold_results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"📁 Создана папка: {results_folder}")
    
    # ============================================
    # ПРИМЕНЯЕМ БИНАРИЗАЦИЮ С РАЗНЫМИ ПОРОГАМИ
    # ============================================
    print("\n🔄 Применяем пороговую бинаризацию...")
    
    # Сохраняем оригинал
    cv2.imwrite(os.path.join(results_folder, '1_original.jpg'), original_image)
    
    # Преобразуем в оттенки серого (для информации)
    gray_image = rgb_to_grayscale_manual(original_image)
    cv2.imwrite(os.path.join(results_folder, '2_grayscale.jpg'), gray_image)
    print("   ✓ Оттенки серого - сохранено")
    
    # Пробуем разные пороги
    thresholds = [80, 127, 180]
    binary_results = []
    
    for thresh in thresholds:
        print(f"   ⏳ Порог {thresh}...")
        binary_rgb, _ = threshold_rgb(original_image, threshold=thresh)
        
        # Сохраняем результат
        filename = f'3_binary_thresh_{thresh}.jpg'
        cv2.imwrite(os.path.join(results_folder, filename), binary_rgb)
        binary_results.append(binary_rgb)
        print(f"   ✅ Порог {thresh} - готово")
    
    # Создаем сравнительное изображение
    h_stack = np.hstack([
        cv2.resize(original_image, (400, 300)),
        cv2.resize(binary_results[0], (400, 300)),
        cv2.resize(binary_results[1], (400, 300)),
        cv2.resize(binary_results[2], (400, 300))
    ])
    
    # Добавляем подписи
    cv2.putText(h_stack, 'Оригинал', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(h_stack, f'Порог {thresholds[0]}', (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(h_stack, f'Порог {thresholds[1]}', (810, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(h_stack, f'Порог {thresholds[2]}', (1210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(os.path.join(results_folder, '4_comparison_all.jpg'), h_stack)
    
    # ============================================
    # ПОКАЗЫВАЕМ РЕЗУЛЬТАТЫ
    # ============================================
    print("\n🖼️ Показываем результаты...")
    
    # Создаем окна
    cv2.namedWindow('Оригинал', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Оттенки серого', cv2.WINDOW_NORMAL)
    cv2.namedWindow(f'Бинаризация (порог {thresholds[0]})', cv2.WINDOW_NORMAL)
    cv2.namedWindow(f'Бинаризация (порог {thresholds[1]})', cv2.WINDOW_NORMAL)
    cv2.namedWindow(f'Бинаризация (порог {thresholds[2]})', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Сравнение всех порогов', cv2.WINDOW_NORMAL)
    
    # Устанавливаем размер окон
    cv2.resizeWindow('Оригинал', 600, 400)
    cv2.resizeWindow('Оттенки серого', 600, 400)
    cv2.resizeWindow(f'Бинаризация (порог {thresholds[0]})', 600, 400)
    cv2.resizeWindow(f'Бинаризация (порог {thresholds[1]})', 600, 400)
    cv2.resizeWindow(f'Бинаризация (порог {thresholds[2]})', 600, 400)
    cv2.resizeWindow('Сравнение всех порогов', 1200, 400)
    
    # Показываем изображения
    cv2.imshow('Оригинал', original_image)
    cv2.imshow('Оттенки серого', gray_image)
    cv2.imshow(f'Бинаризация (порог {thresholds[0]})', binary_results[0])
    cv2.imshow(f'Бинаризация (порог {thresholds[1]})', binary_results[1])
    cv2.imshow(f'Бинаризация (порог {thresholds[2]})', binary_results[2])
    cv2.imshow('Сравнение всех порогов', h_stack)
    
    print("\n✅ Готово!")
    print(f"📁 Все результаты сохранены в: {results_folder}")
    print("\n📋 Созданные файлы:")
    files = os.listdir(results_folder)
    for i, file in enumerate(sorted(files), 1):
        print(f"   {i}. {file}")
    
    print("\n⌨️ Нажмите любую клавишу для закрытия окон...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Открываем папку с результатами
    os.startfile(results_folder)

def quick_threshold_test():
    """
    Быстрый тест бинаризации с интерактивным выбором порога
    """
    print("\n" + "=" * 60)
    print("БЫСТРЫЙ ТЕСТ БИНАРИЗАЦИИ")
    print("=" * 60)
    
    image_path = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\input_image.jpg'
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Изображение не найдено!")
        return
    
    # Преобразуем в оттенки серого
    gray = rgb_to_grayscale_manual(img)
    
    print("\n🔬 Тестируем разные пороги:")
    print("   Окна будут открыты, нажимайте любую клавишу для переключения")
    
    # Показываем оригинал
    cv2.imshow('Оригинал (цветной)', img)
    cv2.imshow('Оттенки серого', gray)
    
    # Показываем бинаризацию с разными порогами
    for thresh in range(50, 251, 50):
        binary = threshold_grayscale(gray, thresh)
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.imshow(f'Порог {thresh}', binary_rgb)
        print(f"   ✓ Порог {thresh}")
    
    print("\n⌨️ Нажмите любую клавишу для закрытия всех окон...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Основной запуск
    main()
    
    # Раскомментируйте для быстрого теста:
    # quick_threshold_test()
