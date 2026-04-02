# Лабораторная работа 4. Сверточная нейронная сеть

Начнем издалека. CNN (Convolutional Neural Network) - сверточная нейронная сеть. Обычно состоит из сверточных (convolution) слоев и pooling (подвыборка) слоев, полносвязный слой и выходной слой.

<img width="866" height="243" alt="image" src="https://github.com/user-attachments/assets/86a47ca8-8cb7-4d98-a4cf-05c096a74ef2" />

## Сверточный слой 
используем фильтры (ядра свертки) которые производят свертку входного изображения в зависимости от его размера. Его гиперпараметры F и S - размер фильтра и шаг свертки. Результатом операции является карта активации или картой признаков. 

<img width="433" height="201" alt="image" src="https://github.com/user-attachments/assets/f47236ce-0e47-471c-a0e0-41d703d80c44" />

## Pooling
Этот слой уменьшает размерность изображения. Наиболее популярные типы pooling: max и average. Max pooling сохраняет наиболее заметные признаки. Average pooling уменьшает размерность карты активации, это довольно специфичный тип pooling, его применяют для LeNet. 

<img width="585" height="193" alt="image" src="https://github.com/user-attachments/assets/849c1e3b-38d5-4198-8d3d-3272dee5e661" />

## Полносвязный слой

<img width="525" height="200" alt="image" src="https://github.com/user-attachments/assets/69b4e883-7be4-43ea-a730-89ef5f7b9b41" />

## Гиперпараметры фильтра (ядра свертки)

Фильтр размера F*F (28*28), применяемый к входу с C каналов (3) представляет собой объем F*F*C. Он применяет свертку с входным размером I*I*C и создает одну карту признаков размером 0*0*1. 

<img width="282" height="198" alt="image" src="https://github.com/user-attachments/assets/1d626761-2469-40bd-b900-aaaebd6f28ea" />

Шаг обозначает количество пикселей, на которое перемещается окно после каждой операции. 

<img width="724" height="85" alt="image" src="https://github.com/user-attachments/assets/105e8f43-c840-4b76-927b-94c9d6ae5294" />

## Размер выхода

<img width="640" height="395" alt="image" src="https://github.com/user-attachments/assets/1b075676-82a9-4e11-a517-76caedd4c815" />

O = (28 -  +  + )/ + 1

1. Загружаем Дата-сет Fashion-MNIST. Выбрала его так как работа будет в коллабе, а размеры этих изображений маленькие (28*28).

<img width="1035" height="191" alt="image" src="https://github.com/user-attachments/assets/4a058979-77f1-4938-a38c-17f0bb020900" />

2. Добавим гауссовского шума и шума типа соль и перец

<img width="1019" height="196" alt="image" src="https://github.com/user-attachments/assets/53366a44-0cde-49d2-96dd-8bc5d449d5dc" />

3. Архитектура нейросети: U-Net + классификатор

<img width="733" height="485" alt="image" src="https://github.com/user-attachments/assets/928a95cd-4ae2-41e1-8f3b-c6bb404df120" />

На вход подается зашумленное изображение RGB (28*28*3). Дальше идут свертки (convolution) и pooling и получаем сжатое представление (7*7*64). Сверточный слой выделяет на изображении характерные примитивы. 

Энкодер сжимает изображение, выделяя признаки. 

<img width="681" height="220" alt="image" src="https://github.com/user-attachments/assets/59253938-25cc-4412-a184-1457851deb15" />

Далее идет классификатор - определяет класс одежды и декодер, восстанавливающий изображение.

Далее сравниваем с эталоном, корректируем веса

Эпох обучения сделано 12

<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/03a7493b-3a26-48cf-a1b5-782f1fd998fa" />


Оцениваем качество на тестовой выборке

<img width="1099" height="339" alt="image" src="https://github.com/user-attachments/assets/63ac75d8-604b-48f6-a51f-fa18eedbdc0a" />

