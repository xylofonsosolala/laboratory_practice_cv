import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma = 1):
    if size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечетным числом")
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size/2
    constant = 1/ (2*np.pi * sigma**2)
    sum_value = 0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            exponent = -(x**2 + y**2)/ (2*sigma**2)
            kernel[i,j]=constant*np.exp(exponent)
            sum_value+=kernel[i,j]
    kernel = kernel / sum_value
    return kernel

def gaussian_filter(image, kernel_size=3, sigma = 1):
    height, width, channels = image.shape
    kernel = gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    filtered_image = np.zeros_like(image, dtype = np.float32)
    padded_image = np.pad(image, pad_width=((pad, pad), (pad, pad), (0,0)), mode = 'edge')
    for i in range(height):
        for j in range(width):
            for c in range (channels):
                window = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                filtered_value = np.sum(window*kernel)
                filtered_image[i,j,c] = filtered_value
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image

def compare_gaussian_parameters():
    image_path = 'C:\\Users\\Admin\\Desktop\\Xenia\\rtkis\\laba\\input_image.jpg'  
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print("Ошибка: Не удалось загрузить изображение")
        return
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    filters = [
        {'size': 3, 'sigma': 0.5, 'name': '3x3, σ=0.5'},
        {'size': 3, 'sigma': 1.0, 'name': '3x3, σ=1.0'},
        {'size': 5, 'sigma': 1.0, 'name': '5x5, σ=1.0'},
        {'size': 5, 'sigma': 2.0, 'name': '5x5, σ=2.0'},
        {'size': 7, 'sigma': 1.5, 'name': '7x7, σ=1.5'},
        {'size': 7, 'sigma': 3.0, 'name': '7x7, σ=3.0'},
    ]
    results = []
    for params in filters:
        print(f" Применяем: {params['name']}")
        filtered = gaussian_filter(original_image, 
                                 kernel_size=params['size'],
                                 sigma=params['sigma'])
        filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        results.append((params['name'], filtered_rgb))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Оригинал
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Оригинал')
    axes[0, 0].axis('off')

    for idx, (name, img) in enumerate(results[:3]):
        axes[0, idx+1].imshow(img)
        axes[0, idx+1].set_title(name)
        axes[0, idx+1].axis('off')
    
    for idx, (name, img) in enumerate(results[3:]):
        axes[1, idx].imshow(img)
        axes[1, idx].set_title(name)
        axes[1, idx].axis('off')

    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

def median_filter(image, kernel_size=3):
    
    height, width, channels = image.shape
    
    if kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечетным числом")
    
    pad = kernel_size // 2
    
    filtered_image = np.zeros_like(image)
    
    # Добавляем отступы к исходному изображению
    # Это позволяет обрабатывать пиксели на границах
    padded_image = np.pad(image, 
                          pad_width=((pad, pad), (pad, pad), (0, 0)),
                          mode='edge')  # режим 'edge' - копирование краевых пикселей
    
    # Проходим по каждому пикселю изображения
    for i in range(height):
        for j in range(width):
            for c in range(channels):  # обрабатываем каждый цветовой канал отдельно
                # Извлекаем окрестность текущего пикселя размером kernel_size x kernel_size
                window = padded_image[i:i + kernel_size, 
                                      j:j + kernel_size, 
                                      c]
                
                # Преобразуем двумерное окно в одномерный массив
                window_flat = window.flatten()
                
                # Сортируем значения в окне
                window_sorted = np.sort(window_flat)
                
                # Находим медиану - центральный элемент отсортированного массива
                median_index = len(window_sorted) // 2
                median_value = window_sorted[median_index]
                
                # Записываем медианное значение в результирующее изображение
                filtered_image[i, j, c] = median_value
    
    return filtered_image




def main():
    # Загружаем изображение
    image_path = 'C:\\Users\\Admin\\Desktop\\Xenia\\rtkis\\laba\\input_image.jpg'  
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print("Ошибка: Не удалось загрузить изображение")
        return

    print(f"Изображение загружено: {original_image.shape[1]}x{original_image.shape[0]}")
    # Применяем медианный фильтр с размером ядра 3x3
    filtered_image_3 = median_filter(original_image, kernel_size=3)
    
    # Применяем медианный фильтр с размером ядра 5x5
    filtered_image_5 = median_filter(original_image, kernel_size=5)
    
    # Отображаем результаты
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Median Filter 3x3', filtered_image_3)
    cv2.imshow('Median Filter 5x5', filtered_image_5)
    
    # Ждем нажатия клавиши для закрытия окон
    print("Нажмите любую клавишу для закрытия окон...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Сохраняем результаты
    cv2.imwrite('filtered_3x3.jpg', filtered_image_3)
    cv2.imwrite('filtered_5x5.jpg', filtered_image_5)

    gaussian_light = gaussian_filter(original_image, kernel_size=3, sigma=0.8)
    
    # Вариант 2: Среднее размытие
    gaussian_medium = gaussian_filter(original_image, kernel_size=5, sigma=1.5)
    
    # Вариант 3: Сильное размытие
    gaussian_strong = gaussian_filter(original_image, kernel_size=7, sigma=2.5)

    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    gaussian_light_rgb = cv2.cvtColor(gaussian_light, cv2.COLOR_BGR2RGB)
    gaussian_medium_rgb = cv2.cvtColor(gaussian_medium, cv2.COLOR_BGR2RGB)
    gaussian_strong_rgb = cv2.cvtColor(gaussian_strong, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_rgb)
    plt.title('Оригинальное изображение')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(gaussian_light_rgb)
    plt.title('Фильтр Гаусса: 3x3, σ=0.8 (слабое размытие)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(gaussian_medium_rgb)
    plt.title('Фильтр Гаусса: 5x5, σ=1.5 (среднее размытие)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(gaussian_strong_rgb)
    plt.title('Фильтр Гаусса: 7x7, σ=2.5 (сильное размытие)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Сохраняем результаты
    print("\n💾 Сохраняем результаты...")
    cv2.imwrite('gaussian_3x3_sigma0.8.jpg', gaussian_light)
    cv2.imwrite('gaussian_5x5_sigma1.5.jpg', gaussian_medium)
    cv2.imwrite('gaussian_7x7_sigma2.5.jpg', gaussian_strong)
    print("✅ Результаты сохранены!")

    print("\n📈 Сравнение различных параметров:")
    compare_gaussian_parameters()

if __name__ == "__main__":
    main()
