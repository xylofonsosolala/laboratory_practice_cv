import cv2
import numpy as np
import os

def dilation(image, kernel_size=3, kernel_shape='square', iterations=1):
    """
    Ручная реализация морфологической операции ДИЛАТАЦИЯ
    """
    # Конвертируем в оттенки серого
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Создаем структурирующий элемент (квадрат)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    
    current_image = gray.copy()
    
    for _ in range(iterations):
        # Добавляем отступы (черный фон - 0)
        padded_image = np.pad(current_image, 
                             pad_width=((pad_h, pad_h), (pad_w, pad_w)),
                             mode='constant', 
                             constant_values=0)
        
        dilated = np.zeros_like(current_image)
        
        for i in range(current_image.shape[0]):
            for j in range(current_image.shape[1]):
                window = padded_image[i:i + kernel_h, j:j + kernel_w]
                # Дилатация: берем МАКСИМУМ в окне
                dilated[i, j] = np.max(window)
        
        current_image = dilated
    
    return current_image

def apply_dilation_to_my_image():
    """
    Применение дилатации к вашему изображению
    """
    print("=" * 60)
    print("МОРФОЛОГИЧЕСКАЯ ОПЕРАЦИЯ: ДИЛАТАЦИЯ")
    print("=" * 60)
    
    # ============================================
    # ПУТЬ К ВАШЕМУ ИЗОБРАЖЕНИЮ - ИЗМЕНИТЕ ПРИ НЕОБХОДИМОСТИ
    # ============================================
    image_path = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\input_image.jpg'
    
    # Загружаем изображение
    print(f"\n📂 Загружаем изображение...")
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print("❌ Ошибка: Не удалось загрузить изображение!")
        return
    
    print(f"✅ Изображение загружено: {original_image.shape[1]}x{original_image.shape[0]}")
    
    # Конвертируем в оттенки серого
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Создаем папку для результатов
    results_folder = r'C:\Users\Admin\Desktop\Xenia\rtkis\laba\dilation_result'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # ============================================
    # ПРИМЕНЯЕМ ДИЛАТАЦИЮ
    # ============================================
    print("\n🔄 Применяем дилатацию...")
    
    # Дилатация 3x3
    print("   - Ядро 3x3...")
    dilated_3x3 = dilation(original_image, kernel_size=3)
    cv2.imwrite(os.path.join(results_folder, 'dilation_3x3.jpg'), dilated_3x3)
    
    # Дилатация 5x5
    print("   - Ядро 5x5...")
    dilated_5x5 = dilation(original_image, kernel_size=5)
    cv2.imwrite(os.path.join(results_folder, 'dilation_5x5.jpg'), dilated_5x5)
    
    # Дилатация 7x7
    print("   - Ядро 7x7...")
    dilated_7x7 = dilation(original_image, kernel_size=7)
    cv2.imwrite(os.path.join(results_folder, 'dilation_7x7.jpg'), dilated_7x7)
    
    # Сохраняем оригинал для сравнения
    cv2.imwrite(os.path.join(results_folder, 'original_gray.jpg'), gray_image)
    cv2.imwrite(os.path.join(results_folder, 'original_color.jpg'), original_image)
    
    # ============================================
    # ПОКАЗЫВАЕМ РЕЗУЛЬТАТЫ
    # ============================================
    print("\n🖼️ Показываем результаты...")
    
    cv2.namedWindow('Оригинал', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Дилатация 3x3', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Дилатация 5x5', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Дилатация 7x7', cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow('Оригинал', 600, 400)
    cv2.resizeWindow('Дилатация 3x3', 600, 400)
    cv2.resizeWindow('Дилатация 5x5', 600, 400)
    cv2.resizeWindow('Дилатация 7x7', 600, 400)
    
    cv2.imshow('Оригинал', gray_image)
    cv2.imshow('Дилатация 3x3', dilated_3x3)
    cv2.imshow('Дилатация 5x5', dilated_5x5)
    cv2.imshow('Дилатация 7x7', dilated_7x7)
    
    print("\n✅ Готово!")
    print(f"📁 Результаты сохранены в: {results_folder}")
    print("\n⌨️ Нажмите любую клавишу для закрытия окон...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Открываем папку с результатами
    os.startfile(results_folder)

if __name__ == "__main__":
    apply_dilation_to_my_image()
