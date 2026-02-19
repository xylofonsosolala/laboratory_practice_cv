import cv2
import numpy as np
import os

def rotate_90_clockwise(image):
    if len(image.shape) == 3:
        h, w, c = image.shape
        rotated = np.zeros((w, h, c), dtype=image.dtype)
        
        for i in range(w): 
            for j in range(h): 
                for k in range(c):
                    rotated[i, j, k] = image[h-1-j, i, k]
    else:
        h, w = image.shape
        rotated = np.zeros((w, h), dtype=image.dtype)
        
        for i in range(w):
            for j in range(h):
                rotated[i, j] = image[h-1-j, i]
    
    return rotated

def rotate_90_counterclockwise(image):
    """
    Поворот изображения на 90 градусов против часовой стрелки
    
    Формула: new[i][j] = old[j][w-1-i]
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
        rotated = np.zeros((w, h, c), dtype=image.dtype)
        
        for i in range(w):
            for j in range(h):
                for k in range(c):
                    rotated[i, j, k] = image[j, w-1-i, k]
    else:
        h, w = image.shape
        rotated = np.zeros((w, h), dtype=image.dtype)
        
        for i in range(w):
            for j in range(h):
                rotated[i, j] = image[j, w-1-i]
    
    return rotated

def rotate_180(image):
    """
    Поворот изображения на 180 градусов
    
    Формула: new[i][j] = old[h-1-i][w-1-j]
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
        rotated = np.zeros((h, w, c), dtype=image.dtype)
        
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    rotated[i, j, k] = image[h-1-i, w-1-j, k]
    else:
        h, w = image.shape
        rotated = np.zeros((h, w), dtype=image.dtype)
        
        for i in range(h):
            for j in range(w):
                rotated[i, j] = image[h-1-i, w-1-j]
    
    return rotated

def rotate_image(image, angle):
    """
    Универсальная функция поворота на углы, кратные 90 градусам
    
    Параметры:
    image - входное изображение
    angle - угол поворота (0, 90, 180, 270, -90, -180, -270)
    
    Возвращает:
    rotated - повернутое изображение
    """
    # Нормализуем угол к диапазону 0-360
    angle = angle % 360
    
    if angle == 0:
        return image.copy()
    elif angle == 90:
        return rotate_90_clockwise(image)
    elif angle == 180:
        return rotate_180(image)
    elif angle == 270:
        # 270 градусов по часовой = 90 против часовой
        return rotate_90_counterclockwise(image)
    else:
        raise ValueError("Угол должен быть кратен 90 градусам (0, 90, 180, 270, -90, -180, -270)")

def apply_rotation_to_my_image():
    """
    Применение поворота к вашему изображению
    """
    print("=" * 70)
    print("ПОВОРОТ ИЗОБРАЖЕНИЯ НА УГЛЫ, КРАТНЫЕ 90 ГРАДУСАМ")
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
    
    h, w, c = original_image.shape
    print(f"✅ Изображение загружено!")
    print(f"   Размер: {w}x{h} пикселей")
    print(f"   Каналов: {c}")
    
    # Создаем папку для результатов
    results_folder = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\rotation_results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"📁 Создана папка: {results_folder}")
    
    # Сохраняем оригинал
    cv2.imwrite(os.path.join(results_folder, '1_original.jpg'), original_image)
    
    # ============================================
    # ПРИМЕНЯЕМ ПОВОРОТЫ
    # ============================================
    print("\n🔄 Применяем повороты...")
    
    rotations = [
        (90, "90° по часовой стрелке"),
        (180, "180°"),
        (270, "270° по часовой (или 90° против)"),
        (-90, "-90° (90° против часовой)")
    ]
    
    rotated_images = []
    
    for angle, description in rotations:
        print(f"   ⏳ Поворот на {description}...")
        
        if angle == -90:
            rotated = rotate_90_counterclockwise(original_image)
            filename = f'2_rotate_270_cw.jpg'  # -90 = 270 cw
        else:
            rotated = rotate_image(original_image, angle)
            filename = f'2_rotate_{angle}_cw.jpg'
        
        # Сохраняем результат
        cv2.imwrite(os.path.join(results_folder, filename), rotated)
        rotated_images.append(rotated)
        
        new_h, new_w = rotated.shape[:2]
        print(f"   ✅ {description} - готово (новый размер: {new_w}x{new_h})")
    
    # ============================================
    # СОЗДАЕМ СРАВНИТЕЛЬНЫЕ ИЗОБРАЖЕНИЯ
    # ============================================
    print("\n🎨 Создаем сравнительные изображения...")
    
    # Подготавливаем изображения для коллажа (все к одинаковой высоте)
    target_height = 300
    
    def resize_for_collage(img, target_h):
        aspect_ratio = img.shape[1] / img.shape[0]
        target_w = int(target_h * aspect_ratio)
        return cv2.resize(img, (target_w, target_h))
    
    orig_resized = resize_for_collage(original_image, target_height)
    rot90_resized = resize_for_collage(rotated_images[0], target_height)
    rot180_resized = resize_for_collage(rotated_images[1], target_height)
    rot270_resized = resize_for_collage(rotated_images[2], target_height)
    
    # Создаем коллаж 2x2
    top_row = np.hstack([orig_resized, rot90_resized])
    bottom_row = np.hstack([rot180_resized, rot270_resized])
    
    # Добавляем разделители
    separator = np.ones((target_height, 10, 3), dtype=np.uint8) * 255
    top_row_with_sep = np.hstack([orig_resized, separator, rot90_resized])
    bottom_row_with_sep = np.hstack([rot180_resized, separator, rot270_resized])
    
    collage = np.vstack([
        top_row_with_sep,
        np.ones((10, top_row_with_sep.shape[1], 3), dtype=np.uint8) * 255,
        bottom_row_with_sep
    ])
    
    # Добавляем подписи
    cv2.putText(collage, 'Оригинал (0°)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(collage, '90°', (orig_resized.shape[1] + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(collage, '180°', (10, target_height + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(collage, '270°', (orig_resized.shape[1] + 30, target_height + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(os.path.join(results_folder, '3_collage_all_rotations.jpg'), collage)
    
    # Создаем анимацию поворота (последовательность)
    rotation_animation = np.hstack([
        resize_for_collage(original_image, 200),
        resize_for_collage(rotated_images[0], 200),
        resize_for_collage(rotated_images[1], 200),
        resize_for_collage(rotated_images[2], 200),
        resize_for_collage(original_image, 200)  # Замыкаем круг
    ])
    
    # Добавляем подписи
    x_positions = [10, 210, 410, 610, 810]
    labels = ['0°', '90°', '180°', '270°', '360°']
    for x, label in zip(x_positions, labels):
        cv2.putText(rotation_animation, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(os.path.join(results_folder, '4_rotation_animation.jpg'), rotation_animation)
    
    # ============================================
    # ПОКАЗЫВАЕМ РЕЗУЛЬТАТЫ
    # ============================================
    print("\n🖼️ Показываем результаты...")
    
    # Создаем окна
    windows = [
        ('Оригинал (0°)', original_image),
        ('Поворот 90°', rotated_images[0]),
        ('Поворот 180°', rotated_images[1]),
        ('Поворот 270°', rotated_images[2]),
        ('Поворот -90°', rotated_images[3]),
        ('Все повороты', collage)
    ]
    
    for name, img in windows:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if name == 'Все повороты':
            cv2.resizeWindow(name, 900, 600)
        else:
            cv2.resizeWindow(name, 500, 400)
        cv2.imshow(name, img)
    
    # ============================================
    # ПРОВЕРКА КОРРЕКТНОСТИ
    # ============================================
    print("\n🔍 Проверка корректности поворотов...")
    
    # Проверка: поворот на 90° дважды = поворот на 180°
    double_90 = rotate_90_clockwise(rotate_90_clockwise(original_image))
    
    # Сравниваем с прямым поворотом на 180°
    direct_180 = rotate_180(original_image)
    
    # Проверяем, что размеры совпадают
    if double_90.shape == direct_180.shape:
        # Проверяем несколько случайных пикселей
        test_passed = True
        for _ in range(10):
            i, j = np.random.randint(0, double_90.shape[0]), np.random.randint(0, double_90.shape[1])
            if not np.array_equal(double_90[i, j], direct_180[i, j]):
                test_passed = False
                break
        
        if test_passed:
            print("   ✅ Проверка пройдена: 90° + 90° = 180°")
        else:
            print("   ⚠️ Проверка не пройдена")
    
    # ============================================
    # ИТОГОВЫЙ ОТЧЕТ
    # ============================================
    print("\n" + "=" * 70)
    print("✅ ПОВОРОТЫ УСПЕШНО ВЫПОЛНЕНЫ!")
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

def quick_rotation_test():
    """
    Быстрый тест поворотов
    """
    print("\n" + "=" * 70)
    print("БЫСТРЫЙ ТЕСТ ПОВОРОТОВ")
    print("=" * 70)
    
    image_path = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\input_image.jpg'
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Изображение не найдено!")
        return
    
    print(f"\n🔄 Тестируем повороты...")
    
    # Создаем окна
    cv2.namedWindow('Оригинал', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Оригинал', 400, 300)
    cv2.imshow('Оригинал', img)
    
    # Поворот на 90°
    rot90 = rotate_90_clockwise(img)
    cv2.namedWindow('90° по часовой', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('90° по часовой', 400, 300)
    cv2.imshow('90° по часовой', rot90)
    
    # Поворот на 180°
    rot180 = rotate_180(img)
    cv2.namedWindow('180°', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('180°', 400, 300)
    cv2.imshow('180°', rot180)
    
    # Поворот на 270°
    rot270 = rotate_90_counterclockwise(img)
    cv2.namedWindow('270° (или -90°)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('270° (или -90°)', 400, 300)
    cv2.imshow('270° (или -90°)', rot270)
    
    print("\n📊 Информация о размерах:")
    print(f"   Оригинал: {img.shape[1]}x{img.shape[0]}")
    print(f"   90°: {rot90.shape[1]}x{rot90.shape[0]}")
    print(f"   180°: {rot180.shape[1]}x{rot180.shape[0]}")
    print(f"   270°: {rot270.shape[1]}x{rot270.shape[0]}")
    
    print("\n⌨️ Нажмите любую клавишу для закрытия...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demonstrate_rotation_sequence():
    """
    Демонстрация последовательности поворотов
    """
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ ПОСЛЕДОВАТЕЛЬНОСТИ ПОВОРОТОВ")
    print("=" * 70)
    
    image_path = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\input_image.jpg'
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Изображение не найдено!")
        return
    
    print("\n🔄 Демонстрируем: 0° → 90° → 180° → 270° → 360°")
    
    # Создаем последовательность
    sequence = [img]
    current = img
    
    for angle in [90, 180, 270, 360]:
        if angle == 90:
            current = rotate_90_clockwise(current)
        elif angle == 180:
            current = rotate_180(current)
        elif angle == 270:
            current = rotate_90_counterclockwise(current)
        elif angle == 360:
            current = img  # Возврат к оригиналу
        
        sequence.append(current)
    
    # Показываем последовательность
    for i, (angle, rotated) in enumerate(zip([0, 90, 180, 270, 360], sequence)):
        cv2.namedWindow(f'{angle}°', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'{angle}°', 400, 300)
        cv2.imshow(f'{angle}°', rotated)
    
    print("\n⌨️ Нажмите любую клавишу для закрытия...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ОСНОВНОЙ ЗАПУСК
    apply_rotation_to_my_image()
    
    # РАСКОММЕНТИРУЙТЕ ДЛЯ ТЕСТОВ:
    # quick_rotation_test()
    # demonstrate_rotation_sequence()
