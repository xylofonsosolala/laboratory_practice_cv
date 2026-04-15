# Лабораторная работа №5
## Обучение нейронной сети на собственном датасете: классификация фациальных обстановок

---

##  Цель работы

Разработать и обучить нейронную сеть для классификации седиментационных обстановок (фаций) по изображениям керна и образцов горных пород. В данной работе решается задача бинарной классификации: **различение озерных (Lakes) и речных (Rivers) отложений**.

---

##  Выбор классов

### Первоначальная попытка

Изначально планировалось классифицировать 4 типа обстановок: **дельта, река, пляж, шельф**. Однако при анализе изображений выяснилось, что визуальные различия между некоторыми классами недостаточно очевидны даже для человека.

| Пляж | Дельта |
|:---:|:---:|
| <img width="150" src="https://github.com/user-attachments/assets/7aad03f9-da0f-4b47-a066-9eac8f9962ba" /> | <img width="150" src="https://github.com/user-attachments/assets/84825269-0b4f-4466-bc58-3db0bb990f92" /> |
| *Визуально неразличимы* | *Визуально неразличимы* |

> **Вывод:** данные классы имеют схожие текстурные характеристики, что делает их классификацию затруднительной без дополнительных геологических признаков.

### Финальные классы: Озёра vs Реки

| **Озёра (Lakes)** | **Реки (Rivers)** |
|:---:|:---:|
| <img width="150" src="https://github.com/user-attachments/assets/1807a863-3e49-4fc5-a0be-d445296c4ff6" /> | <img width="150" src="https://github.com/user-attachments/assets/4f02dc70-59b3-4d69-9f40-69bd59b8c18d" /> |
| <img width="150" src="https://github.com/user-attachments/assets/44e4b380-f416-47dd-9db7-9ee3031452e2" /> | <img width="150" src="https://github.com/user-attachments/assets/0f0b8223-e582-42cf-9881-e4accfbaa9d7" /> |
| <img width="150" src="https://github.com/user-attachments/assets/b081f764-c834-43e4-a02b-1fac4a8e2b08" /> | <img width="150" src="https://github.com/user-attachments/assets/ff95fd94-f0a8-4852-92be-4b1af25e56fe" /> |

#### Ключевые различия:

| Признак |  Озёра |  Реки |
|---------|----------|---------|
| **Слоистость** | Выраженная, **горизонтальная** | Косослоистые структуры |
| **Текстура** | Более однородная | Выраженная **зернистость** |
| **Характер наслоения** | Параллельные слои | Перекрестная слоистость |

---

##  Структура датасета

<img width="467" height="251" alt="image" src="https://github.com/user-attachments/assets/b143606c-b4d0-48a3-845a-04307bc97a6a" />

Внутри val и train лежат соответственно Rivers и Lakes


- **Источник изображений:** скриншоты из учебных материалов по седиментологии
- **Разрешение:** в среднем ~194×344 пикселей
- **Всего изображений:** 140 (70 на класс)

### Использованные учебные материалы:
- Барабошкин — *Практическая седиментология (терригенные коллектора)*
- Алексеев — *Атлас фаций*
- Тугарова, Жуковская — *Атлас типовых фаций*

---

## Предобработка данных

### Параметры обработки

```python
IMG_SIZE = (224, 224)   # Стандартный размер для моделей
BATCH_SIZE = 16         # Размер батча для обучения
```

Аугментация данных
Из-за малого количества данных (всего 70 изображений на класс) была применена аугментация:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Нормализация
    rotation_range=25,           # Повороты
    width_shift_range=0.15,      # Горизонтальные сдвиги
    height_shift_range=0.15,     # Вертикальные сдвиги
    shear_range=0.15,            # Сдвиги
    zoom_range=0.15,             # Масштабирование
    horizontal_flip=True,        # Горизонтальное отражение
    vertical_flip=True,          # Вертикальное отражение
    brightness_range=[0.8, 1.2], # Изменение яркости
    fill_mode='nearest'
)
```

Выбор модели: MobileNetV2 (предобучена на ImageNet)

```python
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
```

Выбрала усреднение признаков так как если слоистости нет, а есть просто шум, то шум усилится при max pooling. 

Optimizer: Adam
Adam (Adaptive Moment Estimation) - это алгоритм оптимизации, который обновляет веса нейронной сети во время обучения

Подробнее тут
https://www.geeksforgeeks.org/deep-learning/adam-optimizer/

активации

 ## Метрики оценки
Основная метрика: F1-score (weighted)

Accuracy может вводить в заблуждение при дисбалансе классов, поэтому выбран F1-score

## Параметры обучения

50 эпох с ранней остановкой

Финальные метрики:
  • Precision: 0.8646
  • Recall: 0.8571
  • F1-score: 0.8564
  • Accuracy: 0.8571 (85.71%)

  ## Матрица ошибок

  <img width="1195" height="518" alt="image" src="https://github.com/user-attachments/assets/65219e86-1868-4fbd-9e01-89325d9a2739" />

  Итоговые графики обучения

  | Loss | F1 |
|:---:|:---:|
| <img width="605" height="376" alt="image" src="https://github.com/user-attachments/assets/ffe8ebbf-b667-48a6-b6f6-e13370ce86e2" />
 | <img width="621" height="383" alt="image" src="https://github.com/user-attachments/assets/4f779436-b5b0-455f-9bd5-c0f27c84c7ca" />
|

