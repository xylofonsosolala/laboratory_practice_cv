# Лабораторная работа 5. Обучение нейронной сети на своем датасете
Для начало было выбрано обучать модель классифицировать фациальные обстановки осадконакопления: сравнивать озеры и реки. 
Были предприняты попытки классифицировать другие обстановки, однако я их отличить не могу, например

<img width="91" height="386" alt="21" src="https://github.com/user-attachments/assets/7aad03f9-da0f-4b47-a066-9eac8f9962ba" />

Это пляж

<img width="154" height="541" alt="23" src="https://github.com/user-attachments/assets/84825269-0b4f-4466-bc58-3db0bb990f92" />

А это дельта. Unreal

Поэтому были выбраны озера и реки. 

<img width="194" height="344" alt="1" src="https://github.com/user-attachments/assets/1807a863-3e49-4fc5-a0be-d445296c4ff6" /> 
<img width="194" height="344" alt="2" src="https://github.com/user-attachments/assets/44e4b380-f416-47dd-9db7-9ee3031452e2" />
<img width="194" height="344" alt="3" src="https://github.com/user-attachments/assets/b081f764-c834-43e4-a02b-1fac4a8e2b08" />

Это озера

<img width="100" height="283" alt="3" src="https://github.com/user-attachments/assets/4f02dc70-59b3-4d69-9f40-69bd59b8c18d" />
<img width="100" height="283" alt="4" src="https://github.com/user-attachments/assets/0f0b8223-e582-42cf-9881-e4accfbaa9d7" />
<img width="100" height="283" alt="5" src="https://github.com/user-attachments/assets/ff95fd94-f0a8-4852-92be-4b1af25e56fe" />

Это реки

Хорошо видно что у озер более выражена слоистость, причем преобладает горизонтальная
А у рек выраженная зернистность, косослоистые структуры

## Подготовка данных

Были созданы папка train и val с изображениями Lakes (озер) и Rivers (рек). Все изображения представляют собой скриншоты из учебных материалов по седиментологии. Разрешение изображений варьируется, в среднем ~194×344 пикселей. 

Учебники: Барабошкин - Практическая седиментология (терригенные коллектора)
Алексеев - Атлас фаций
Тугарова, Жуковская - Атлас типовых фаций

### Параметры обработки изображений
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

Без аугментации к сожалению не обошлось. 70 фотографий крайне мало для обучения

Использована предобученная модель MobileNetV2

## Метрики оценки

Основная метрика - F1-score (weighted), так как accuracy может вводить в заблуждение при дисбалансе классов.
Конечные метрики на Озера и Реки соответственно:
Precision (weighted): 0,81 и 0,92
Recall (weighted):    0,93 и 0,79
F1-score (weighted):  0,87 и 0,85
Accuracy:             0.8571

## Процесс обучения

Epochs: 50 (с ранней остановкой)
Learning rate: 0.0005

Для обучения выбрана предобученная модель MobileNetV2. Подробнее можно почитать https://www.geeksforgeeks.org/computer-vision/what-is-mobilenet-v2/

На первых этапах была обнаружена проблема: при объединении train и val данных модель показывала 100% точность, но это было следствием утечки - одни и те же изображения присутствовали в обеих выборках.

## Матрица ошибок

<img width="1101" height="459" alt="image" src="https://github.com/user-attachments/assets/bd4d5af1-2863-49f3-8e05-5309785c4a7f" />

Модель успешно различает озера и реки, однако ошибки возникают. 
Думаю это следствие той самой аугментации: как было ранее сказано у рек косая слоистость, но при повороте озер получается тоже косая слоистость.

<img width="1081" height="424" alt="image" src="https://github.com/user-attachments/assets/34832082-2636-4279-9dcf-ed8d44a90850" />

Ну и результат на графиках

<img width="368" height="293" alt="image" src="https://github.com/user-attachments/assets/5a592487-81d3-4d39-8543-f1282f3a3297" />
<img width="550" height="339" alt="image" src="https://github.com/user-attachments/assets/26c7ec2a-6e13-4292-8e96-cbbc3b83ac20" />


