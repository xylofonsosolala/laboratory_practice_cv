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

  x, y - координаты относительно центра ядра, sigma - стандартное отклонение, 1/(2*pi*sigma^2) - нормировочный коэффициент

Фильтр Гаусса - это линейный фильтр, который сглаживает изображение путем усреднения пикселей с весами, определяемыми функцией Гаусса. Пиксели, расположенные ближе к центру, получают больший вес, а дальние - меньший. Плохо работает на соль и перец, по большей степени размывает изображение
  ![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/median.png?raw=true)

## 2. Морфологические операции
### 2.1. Эрозия (уменьшение объектов)

Эрозия "сужает" границы объектов. Пиксель становится белым (1) только если оба пикселя под структурным элементом белые

<img width="297" height="371" alt="укщышщ" src="https://github.com/user-attachments/assets/2222c8bb-897c-4bfe-890c-e06b6394d721" />


### 2.2. Дилатация (увеличение)

![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/erosiondelation.png?raw=true)

Дилатация "расширяет" границы объектов. Пиксель становится белым (1) если хотя бы один пиксель под структурным элементом белый.

<img width="262" height="365" alt="птгащпващптващ" src="https://github.com/user-attachments/assets/b3ff3f0f-0de9-4f03-9b8b-83c3c141a6e4" />

## 3. Пороговая бинаризация

- Пороги: 80, 127
![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/binarisation.png?raw=true)

## 4. Выравнивание гистограммы

Выравнивание гистограммы - это метод улучшения контрастности изображения путем перераспределения интенсивности пикселей так, чтобы гистограмма стала более равномерной.
![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/hystogram.png?raw=true)

## 5. Поворот изображения на угол, кратный 90 градусов
Поворот на углы, кратные 90 градусам.
- Углы: 0°, 90°, 180°, 270°
![](https://github.com/xylofonsosolala/laboratory_practice_cv/blob/main/rotation.png?raw=true)
