import cv2
import numpy as np
import os

def create_structuring_element(shape='square', size=3):
    if size % 2 == 0:
        raise ValueError("Размер должен быть нечетным числом")
    
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    if shape == 'square':
        # Квадрат - все единицы
        kernel[:, :] = 1
        
    elif shape == 'cross':
        # Крест - центральная строка и столбец
        kernel[center, :] = 1
        kernel[:, center] = 1
        
    elif shape == 'circle':
        # Круг (приближение)
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist <= center:
                    kernel[i, j] = 1
                    
    elif shape == 'diamond':
        # Ромб
        for i in range(size):
            for j in range(size):
                if abs(i - center) + abs(j - center) <= center:
                    kernel[i, j] = 1
    
    return kernel

def dilation(image, kernel_size=3, kernel_shape='square', iterations=1):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Создаем структурирующий элемент
    kernel = create_structuring_element(kernel_shape, kernel_size)
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    
    current_image = gray.copy()
    
    for _ in range(iterations):
        padded_image = np.pad(current_image, 
                             pad_width=((pad_h, pad_h), (pad_w, pad_w)),
                             mode='constant', 
                             constant_values=0)
        
        dilated = np.zeros_like(current_image)
        
        for i in range(current_image.shape[0]):
            for j in range(current_image.shape[1]):
                # Извлекаем окно
                window = padded_image[i:i + kernel_h, j:j + kernel_w]
                # Применяем структурирующий элемент
                masked_window = window[kernel == 1]
                
                if len(masked_window) > 0:
                    dilated[i, j] = np.max(masked_window)
        
        current_image = dilated
    
    return current_image


def erosion(image, kernel_size=3, kernel_shape='square', iterations=1):
    # Если изображение цветное, конвертируем в оттенки серого
    if len(image.shape) == 3:
        if image.shape[2] == 3 or image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:,:,0]
    else:
        gray = image.copy()
    
    # Создаем структурирующий элемент
    kernel = create_structuring_element(kernel_shape, kernel_size)
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    
    # Копируем изображение для обработки
    current_image = gray.copy()
    
    for _ in range(iterations):
        # Добавляем отступы
        padded_image = np.pad(current_image, 
                             pad_width=((pad_h, pad_h), (pad_w, pad_w)),
                             mode='constant', 
                             constant_values=255)  # Для эрозии используем белый фон
        
        # Создаем результирующее изображение
        eroded = np.zeros_like(current_image)
        
        # Применяем эрозию
        for i in range(current_image.shape[0]):
            for j in range(current_image.shape[1]):
                # Извлекаем окно
                window = padded_image[i:i + kernel_h, j:j + kernel_w]
                
                # Применяем структурирующий элемент
                masked_window = window[kernel == 1]
                
                if len(masked_window) > 0:
                    # Эрозия: берем МИНИМУМ в окне
                    eroded[i, j] = np.min(masked_window)
        
        current_image = eroded
    
    return current_image

def erosion_binary(image, kernel_size=3, kernel_shape='square', threshold=127):
    # Бинаризация изображения
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Пороговая обработка
    binary = np.zeros_like(gray)
    binary[gray > threshold] = 255
    
    # Создаем структурирующий элемент
    kernel = create_structuring_element(kernel_shape, kernel_size)
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    
    # Добавляем отступы
    padded = np.pad(binary, 
                   pad_width=((pad_h, pad_h), (pad_w, pad_w)),
                   mode='constant', 
                   constant_values=0)
    
    # Результат
    eroded = np.zeros_like(binary)
    
    # Применяем эрозию для бинарного изображения
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            window = padded[i:i + kernel_h, j:j + kernel_w]
            # Проверяем, все ли пиксели под ядром белые
            if np.all(window[kernel == 1] == 255):
                eroded[i, j] = 255
    
    return eroded

def visualize_morphology_erosion():
    print("=" * 60)
    print("МОРФОЛОГИЧЕСКАЯ ОПЕРАЦИЯ: ЭРОЗИЯ")
    print("=" * 60)
    
    # Создаем тестовое изображение
    test_img = np.zeros((300, 400), dtype=np.uint8)
    
    # Рисуем различные фигуры
    cv2.rectangle(test_img, (50, 50), (150, 150), 255, -1)  # Квадрат
    cv2.circle(test_img, (250, 100), 50, 255, -1)  # Круг
    cv2.rectangle(test_img, (50, 200), (150, 280), 255, -1)  # Прямоугольник
    cv2.circle(test_img, (250, 240), 40, 255, -1)  # Маленький круг
    
    # Добавляем шум
    for _ in range(500):
        x, y = np.random.randint(0, 400), np.random.randint(0, 300)
        test_img[y, x] = 255
    
    # Создаем папку для результатов
    results_folder = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\morphology_results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Сохраняем оригинал
    cv2.imwrite(os.path.join(results_folder, '01_original.jpg'), test_img)
    
    # 1. Демонстрация разных размеров ядра
    print("\n📊 1. Эрозия с разными РАЗМЕРАМИ ядра (квадрат):")
    
    sizes = [3, 5, 7, 9]
    results_sizes = []
    
    for size in sizes:
        eroded = erosion(test_img, kernel_size=size, kernel_shape='square')
        cv2.imwrite(os.path.join(results_folder, f'02_erosion_square_{size}x{size}.jpg'), eroded)
        results_sizes.append(eroded)
        print(f"   ✓ Ядро {size}x{size} - применено")
    
    # 2. Демонстрация разных форм ядра
    print("\n🔷 2. Эрозия с разными ФОРМАМИ ядра (размер 5x5):")
    
    shapes = ['square', 'cross', 'circle', 'diamond']
    results_shapes = []
    
    for shape in shapes:
        eroded = erosion(test_img, kernel_size=5, kernel_shape=shape)
        cv2.imwrite(os.path.join(results_folder, f'03_erosion_5x5_{shape}.jpg'), eroded)
        results_shapes.append(eroded)
        print(f"   ✓ Форма '{shape}' - применена")
    
    # 3. Демонстрация нескольких итераций
    print("\n🔄 3. Эрозия с разным количеством ИТЕРАЦИЙ (ядро 3x3):")
    
    iterations = [1, 2, 3, 5]
    results_iters = []
    
    for i, iter_count in enumerate(iterations):
        eroded = erosion(test_img, kernel_size=3, kernel_shape='square', iterations=iter_count)
        cv2.imwrite(os.path.join(results_folder, f'04_erosion_3x3_iter{iter_count}.jpg'), eroded)
        results_iters.append(eroded)
        print(f"   ✓ {iter_count} итераций - применено")
    
    # 4. Сравнение бинарной и полутоновой эрозии
    print("\n⚪ 4. Сравнение бинарной и полутоновой эрозии:")
    
    # Создаем градиентное изображение
    gradient = np.zeros((300, 400), dtype=np.uint8)
    for i in range(300):
        for j in range(400):
            gradient[i, j] = (i + j) // 3
    
    cv2.imwrite(os.path.join(results_folder, '05_gradient_original.jpg'), gradient)
    
    erosion_gray = erosion(gradient, kernel_size=5, kernel_shape='square')
    erosion_bin = erosion_binary(gradient, kernel_size=5, kernel_shape='square')
    
    cv2.imwrite(os.path.join(results_folder, '06_gradient_erosion_gray.jpg'), erosion_gray)
    cv2.imwrite(os.path.join(results_folder, '07_gradient_erosion_binary.jpg'), erosion_bin)
    
    print("   ✓ Полутоновая эрозия - применена")
    print("   ✓ Бинарная эрозия - применена")
    
    # 5. Визуализация структурирующих элементов
    print("\n🔲 5. Визуализация структурирующих элементов:")
    
    viz_kernels = []
    for shape in shapes:
        kernel = create_structuring_element(shape, 11) * 255
        # Увеличиваем для наглядности
        kernel_viz = cv2.resize(kernel, (100, 100), interpolation=cv2.INTER_NEAREST)
        kernel_viz = cv2.applyColorMap(kernel_viz, cv2.COLORMAP_VIRIDIS)
        cv2.putText(kernel_viz, shape, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        viz_kernels.append(kernel_viz)
    
    kernels_grid = np.hstack(viz_kernels)
    cv2.imwrite(os.path.join(results_folder, '08_structuring_elements.jpg'), kernels_grid)
    print("   ✓ Структурирующие элементы - визуализированы")
    
    # Показываем результаты в окнах OpenCV
    print("\n🖼️ Показываем результаты...")
    
    cv2.namedWindow('Оригинал', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Оригинал', 400, 300)
    cv2.imshow('Оригинал', test_img)
    
    # Показываем результаты с разными размерами
    for i, (size, result) in enumerate(zip(sizes, results_sizes)):
        cv2.namedWindow(f'Эрозия {size}x{size}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Эрозия {size}x{size}', 400, 300)
        cv2.imshow(f'Эрозия {size}x{size}', result)
    
    print("\n⌨️ Нажмите любую клавишу для закрытия окон...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"\n📁 Все результаты сохранены в: {results_folder}")
    os.startfile(results_folder)
    
    return results_folder

def apply_erosion_to_image():
    """
    Применение эрозии к реальному изображению
    """
    print("\n" + "=" * 60)
    print("ПРИМЕНЕНИЕ ЭРОЗИИ К ИЗОБРАЖЕНИЮ")
    print("=" * 60)
    
    # Путь к изображению
    image_path = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\input_image.jpg'
    
    # Загружаем изображение
    img = cv2.imread(image_path)
    
    if img is None:
        print("❌ Изображение не найдено. Использую тестовое...")
        # Создаем тестовое изображение
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.circle(img, (450, 200), 80, (255, 255, 255), -1)
        # Добавляем текст
        cv2.putText(img, 'EROSION TEST', (200, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Добавляем шум
        for _ in range(2000):
            x, y = np.random.randint(0, 600), np.random.randint(0, 400)
            img[y, x] = [255, 255, 255]
    
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Создаем папку для результатов
    results_folder = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\erosion_results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Сохраняем оригинал
    cv2.imwrite(os.path.join(results_folder, 'original.jpg'), img)
    cv2.imwrite(os.path.join(results_folder, 'original_gray.jpg'), gray)
    
    # Применяем эрозию с разными параметрами
    print("\n🔄 Применяем эрозию к изображению...")
    
    # 1. Квадратное ядро разных размеров
    for size in [3, 5, 7]:
        eroded = erosion(gray, kernel_size=size, kernel_shape='square')
        cv2.imwrite(os.path.join(results_folder, f'erosion_square_{size}.jpg'), eroded)
        print(f"   ✓ Квадрат {size}x{size}")
    
    # 2. Разные формы ядра
    for shape in ['square', 'cross', 'circle']:
        eroded = erosion(gray, kernel_size=5, kernel_shape=shape)
        cv2.imwrite(os.path.join(results_folder, f'erosion_5x5_{shape}.jpg'), eroded)
        print(f"   ✓ Форма '{shape}'")
    
    # 3. Бинарная эрозия
    eroded_binary = erosion_binary(gray, kernel_size=3)
    cv2.imwrite(os.path.join(results_folder, 'erosion_binary_3x3.jpg'), eroded_binary)
    print(f"   ✓ Бинарная эрозия")
    
    # 4. Много итераций
    for iter_count in [2, 3]:
        eroded = erosion(gray, kernel_size=3, iterations=iter_count)
        cv2.imwrite(os.path.join(results_folder, f'erosion_iter{iter_count}.jpg'), eroded)
        print(f"   ✓ {iter_count} итерации")
    
    print(f"\n✅ Все результаты сохранены в: {results_folder}")
    
    # Показываем результаты
    cv2.namedWindow('Оригинал (цветной)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Оригинал (серый)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Эрозия 3x3', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Эрозия 5x5', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Эрозия 7x7', cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow('Оригинал (цветной)', 400, 300)
    cv2.resizeWindow('Оригинал (серый)', 400, 300)
    cv2.resizeWindow('Эрозия 3x3', 400, 300)
    cv2.resizeWindow('Эрозия 5x5', 400, 300)
    cv2.resizeWindow('Эрозия 7x7', 400, 300)
    
    cv2.imshow('Оригинал (цветной)', img)
    cv2.imshow('Оригинал (серый)', gray)
    cv2.imshow('Эрозия 3x3', erosion(gray, 3))
    cv2.imshow('Эрозия 5x5', erosion(gray, 5))
    cv2.imshow('Эрозия 7x7', erosion(gray, 7))
    
    print("\n⌨️ Нажмите любую клавишу для закрытия окон...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    os.startfile(results_folder)
    
    return results_folder

def compare_erosion_dilation(image):
    """
    Сравнение эрозии и дилатации
    """
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ: ЭРОЗИЯ vs ДИЛАТАЦИЯ")
    print("=" * 70)
    
    # Конвертируем в оттенки серого
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        compare_folder = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\erosion_vs_dilation'
    if not os.path.exists(compare_folder):
        os.makedirs(compare_folder)
    
    # Применяем операции
    sizes = [3, 5, 7]
    
    for size in sizes:
        eroded = erosion(gray, kernel_size=size, kernel_shape='square')
        dilated = dilation(gray, kernel_size=size, kernel_shape='square')
        
        # Создаем сравнение
        comparison = np.hstack([
            cv2.resize(gray, (400, 300)),
            cv2.resize(eroded, (400, 300)),
            cv2.resize(dilated, (400, 300))
            
])
        
        # Добавляем подписи
        cv2.putText(comparison, f'Оригинал', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(comparison, f'Эрозия {size}x{size}', (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(comparison, f'Дилатация {size}x{size}', (810, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        cv2.imwrite(os.path.join(compare_folder, f'compare_{size}x{size}.jpg'), comparison)
        print(f"   ✓ Сравнение для ядра {size}x{size} - сохранено")
    
    return compare_folder


def apply_dilation_to_my_image():
    """
    Применение дилатации к вашему изображению
    """
    print("=" * 70)
    print("МОРФОЛОГИЧЕСКАЯ ОПЕРАЦИЯ: ДИЛАТАЦИЯ - ВАШЕ ИЗОБРАЖЕНИЕ")
    print("=" * 70)
    
    image_path = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\input_image.jpg'
    
    # Проверяем существование файла
    if not os.path.exists(image_path):
        print(f"❌ Файл не найден: {image_path}")
        return
    
    # Загружаем изображение
    print(f"\n📂 Загружаем изображение: {image_path}")
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print("❌ Не удалось загрузить изображение!")
        return
    
    print(f"✅ Изображение загружено успешно!")
    print(f"   Размер: {original_image.shape[1]}x{original_image.shape[0]} пикселей")
    
    # Создаем папку для результатов дилатации
    results_folder = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\dilation_my_image'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"📁 Создана папка: {results_folder}")
    
    # Сохраняем оригинал
    cv2.imwrite(os.path.join(results_folder, '0_original_color.jpg'), original_image)
    
    # Конвертируем в оттенки серого
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(results_folder, '0_original_gray.jpg'), gray_image)
    
    print("\n" + "=" * 70)
    print("🔄 ПРИМЕНЯЕМ ДИЛАТАЦИЮ С РАЗНЫМИ ПАРАМЕТРАМИ")
    print("=" * 70)
    
    # ============================================
    print("\n📊 1. ДИЛАТАЦИЯ - РАЗНЫЙ РАЗМЕР ЯДРА (квадрат):")
    
    sizes = [3, 5, 7, 9, 11]
    dilation_results = []
    for size in sizes:
        print(f"   ⏳ Применяем ядро {size}x{size}...")
        dilated = dilation(original_image, kernel_size=size, kernel_shape='square')
        
        # Сохраняем результат
        filename = f'1_dilation_square_{size}x{size}.jpg'
        cv2.imwrite(os.path.join(results_folder, filename), dilated)
        
        # Создаем сравнение
        comparison = np.hstack([
            cv2.resize(gray_image, (400, 300)),
            cv2.resize(dilated, (400, 300))
        ])
        cv2.imwrite(os.path.join(results_folder, f'1_comparison_{size}x{size}.jpg'), comparison)
        
        dilation_results.append(dilated)
        print(f"   ✅ {size}x{size} - готово")
        print("\n🔷 2. ДИЛАТАЦИЯ - РАЗНАЯ ФОРМА ЯДРА (5x5):")
    
    shapes = ['square', 'cross', 'circle', 'diamond', 'vertical', 'horizontal']
    shape_names = ['Квадрат', 'Крест', 'Круг', 'Ромб', 'Вертикаль', 'Горизонталь']
    shape_results = []
    
    for shape, name in zip(shapes, shape_names):
        print(f"   ⏳ Применяем форму '{name}'...")
        dilated = dilation(original_image, kernel_size=5, kernel_shape=shape)
        
        # Сохраняем результат
        filename = f'2_dilation_5x5_{shape}.jpg'
        cv2.imwrite(os.path.join(results_folder, filename), dilated)
        
        # Визуализируем структурирующий элемент
        kernel = create_structuring_element(shape, 5) * 255
        kernel_viz = cv2.resize(kernel, (100, 100), interpolation=cv2.INTER_NEAREST)
        kernel_viz = cv2.applyColorMap(kernel_viz, cv2.COLORMAP_VIRIDIS)

        cv2.putText(kernel_viz, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(results_folder, f'2_kernel_{shape}.jpg'), kernel_viz)
        
        shape_results.append(dilated)
        print(f"   ✅ '{name}' - готово")
    
    # ============================================
    # 3. ДИЛАТАЦИЯ С РАЗНЫМ КОЛИЧЕСТВОМ ИТЕРАЦИЙ
    # ============================================
    print("\n🔄 3. ДИЛАТАЦИЯ - РАЗНОЕ КОЛИЧЕСТВО ИТЕРАЦИЙ (3x3):")
    
    iterations = [1, 2, 3, 5]
    iter_results = []
    
    for iter_count in iterations:
        print(f"   ⏳ Применяем {iter_count} итераций...")
        dilated = dilation(original_image, kernel_size=3, kernel_shape='square', iterations=iter_count)
        
        # Сохраняем результат
        filename = f'3_dilation_3x3_iter{iter_count}.jpg'

        cv2.imwrite(os.path.join(results_folder, filename), dilated)
        
        iter_results.append(dilated)
        print(f"   ✅ {iter_count} итераций - готово")
    
    # ============================================
    # 4. БИНАРНАЯ ДИЛАТАЦИЯ
    # ============================================
    print("\n⚪ 4. БИНАРНАЯ ДИЛАТАЦИЯ:")
    
    thresholds = [80, 127, 200]
    
    for thresh in thresholds:
        print(f"   ⏳ Порог бинаризации: {thresh}...")
        dilated_bin = dilation_binary(original_image, kernel_size=3, threshold=thresh)
        
        # Сохраняем результат
        filename = f'4_dilation_binary_thresh{thresh}.jpg'
        cv2.imwrite(os.path.join(results_folder, filename), dilated_bin)
        

if __name__ == "__main__":
    # Демонстрация эрозии на тестовых фигурах
    visualize_morphology_erosion()
    
    # Применение к реальному изображению
    apply_erosion_to_image()
    
    print("\n" + "=" * 60)
    print("✅ ВСЕ ОПЕРАЦИИ ЭРОЗИИ ВЫПОЛНЕНЫ УСПЕШНО!")
    print("=" * 60)
