# Лабораторная работа 1
## 1. Фильтры
### 1.1. Фильтр медианный
Реализация медианного фильтра для подавления импульсного шума.
- Размеры ядра: 3x3, 5x5
<img width="447" height="243" alt="image" src="https://github.com/user-attachments/assets/62133ce3-6fd3-4149-87db-aa06e203fb2b" />

Медианный фильтр проходит ядром по всему изображению, собираются значения всех пикселей внутри окна, сортируются в порядке возрастания, медианное значение ставится в центр, хорошо отрабатывает для шума соль и перец
![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/real_median.png?raw=true)

### 1.2. Фильтр Гаусса
- Размеры ядра: 3x3, 7x7
- Параметр sigma: 0.8, 2.5
  
  <img width="653" height="369" alt="image" src="https://github.com/user-attachments/assets/378b2c76-22ac-459f-96ae-b614dc38c617" />

Фильтр Гаусса - это линейный фильтр, который сглаживает изображение путем усреднения пикселей с весами, определяемыми функцией Гаусса. Пиксели, расположенные ближе к центру, получают больший вес, а дальние - меньший. Плохо работает на соль и перец, по большей степени размывает изображение
  ![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/median.png?raw=true)

## 2. Морфологические операции
### 2.1. Эрозия (уменьшение объектов)

### 2.2. Дилатация (увеличение)

![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/erosiondelation.png?raw=true)

## 3. Пороговая бинаризация

- Пороги: 80, 127
![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/binarisation.png?raw=true)

## 4. Выравнивание гистограммы
![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/hystogram.png?raw=true)

## 5. Поворот изображения на угол, кратный 90 градусов
Поворот на углы, кратные 90 градусам.
- Углы: 0°, 90°, 180°, 270°
![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/rotation.png?raw=true)
