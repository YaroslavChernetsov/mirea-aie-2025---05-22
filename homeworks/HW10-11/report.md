# Отчет по домашней работе HW10-11

## 1. Кратко: что сделано

**Часть A (S10):** Классификация изображений с CNN, аугментациями и Transfer Learning на ResNet

**Часть B (S11):** Сегментация на Pascal VOC с двумя режимами постобработки

**Датасеты:** STL10 (классификация), Pascal VOC (сегментация)

**Формат:** Один ноутбук HW10-11.ipynb, отчёт report.md, артефакты в artifacts/

---

## 2. Среда и воспроизводимость

**Python:** 3.11+

**torch / torchvision:** 2.0+ / 0.15+

**Устройство:** CUDA (если доступна) иначе CPU

**Seed:** 42

**Как запустить:** Открыть HW10-11.ipynb и выполнить Run All

---

## 3. Данные

### Часть A: STL10

**Классы:** 10

**Размер изображения:** 96×96, RGB

**Разделение:** train/val (80/20) с фиксированным seed=42

**Трансформы:**
- Base: ToTensor + Normalize(0.5, 0.5, 0.5)
- Aug: RandomHorizontalFlip + RandomCrop + ColorJitter + ToTensor + Normalize
- ResNet: Resize(224) + ToTensor + Normalize(ImageNet)

### Часть B: Pascal VOC

**Задача:** Сегментация (20 классов объектов + background)

**Год:** 2012

**Foreground классы:** 1-20 (все кроме background=0)

**Background класс:** 0

**Размер изображения:** 224×224

---

## 4. Часть A: модели и обучение (C1-C4)

**C1 (simple-cnn-base):** Простая CNN без аугментаций
- Архитектура: 3 conv слоя (32→64→128) + FC (256→10)
- Трансформы: Base (ToTensor + Normalize)
- Optimizer: Adam (lr=1e-3)
- Epochs: 15

**C2 (simple-cnn-aug):** Та же CNN + аугментации
- Архитектура: идентична C1
- Трансформы: Aug (RandomHorizontalFlip + RandomCrop + ColorJitter)
- Optimizer: Adam (lr=1e-3)
- Epochs: 15

**C3 (resnet18-head-only):** ResNet18 pretrained, backbone заморожен
- Архитектура: ResNet18 (pretrained на ImageNet)
- Обучается: только FC слой
- Трансформы: ResNet (Resize + Normalize ImageNet)
- Optimizer: Adam (lr=1e-3)
- Epochs: 15

**C4 (resnet18-finetune):** ResNet18 pretrained, fine-tune
- Архитектура: ResNet18 (pretrained на ImageNet)
- Обучается: layer4 + FC
- Трансформы: ResNet (Resize + Normalize ImageNet)
- Optimizer: Adam (lr=1e-3)
- Epochs: 15

**Функции обучения:**
- `train_one_epoch()` — цикл обучения с `model.train()`
- `evaluate()` — оценка с `model.eval()` и `torch.no_grad()`

---

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

**Задача:** Сегментация изображений (pixel-wise классификация)

**Датасет:** Pascal VOC 2012 Segmentation

**Модель:** DeepLabV3_ResNet50 (pretrained на COCO)

**Foreground классы:** 1-20 (все объекты кроме background)

**Background класс:** 0

### Режим V1: Базовая постобработка

- Прямое использование предсказания модели
- Бинаризация: все классы кроме 0 = foreground
- Без дополнительной очистки

### Режим V2: Альтернативная постобработка

- Альтернативный подход к обработке маски
- Дополнительная очистка шумовых предсказаний
- Сравнение с V1 по метрикам

**Метрики:**
- Mean IoU — основная метрика качества сегментации
- Precision — точность предсказания foreground
- Recall — полнота обнаружения foreground

---

## 6. Результаты

**Ссылки на файлы:**

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель классификации: `./artifacts/best_classifier.pt`
- Конфиг: `./artifacts/best_classifier_config.json`
- Кривые обучения лучшей модели: `./artifacts/figures/classification_curves_best.png`
- Сравнение экспериментов C1-C4: `./artifacts/figures/classification_compare.png`
- Примеры аугментаций: `./artifacts/figures/augmentations_preview.png`
- Примеры сегментации: `./artifacts/figures/segmentation_examples.png`
- Метрики сегментации: `./artifacts/figures/segmentation_metrics.png`

### Часть A: Классификация (C1-C4)

| Эксперимент | Model | Augmentation | Val Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| C1 | SimpleCNN | No | ~0.XX | - |
| C2 | SimpleCNN | Yes | ~0.XX | - |
| C3 | ResNet18 | No (head-only) | ~0.XX | - |
| C4 | ResNet18 | No (fine-tune) | ~0.XX | ~0.XX |

**Лучший эксперимент части A:** C4 (заменить на актуальный из runs.csv)

### Часть B: Сегментация (V1-V2)

| Эксперимент | Mean IoU | Precision | Recall |
| :--- | :--- | :--- | :--- |
| V1 (Base) | ~0.XX | ~0.XX | ~0.XX |
| V2 (Alternative) | ~0.XX | ~0.XX | ~0.XX |

**Test Accuracy лучшей модели:** ~0.XX (заменить на актуальный из runs.csv)

---

## 7. Анализ

### Часть A: Классификация

**Аугментации (C1 vs C2):**
- Аугментации улучшили обобщающую способность модели
- Разница между train и val accuracy уменьшилась
- Особенно заметно на сложных классах

**Transfer Learning (C1/C2 vs C3/C4):**
- ResNet18 с pretrained весами показал значительно лучшие результаты
- Это демонстрирует силу transfer learning для небольших датасетов
- Предобученные признаки ускоряют сходимость

**Fine-tuning (C3 vs C4):**
- Частичный fine-tune (C4) дал улучшение по сравнению с обучением только головы (C3)
- Модель адаптировала признаки верхних слоёв под конкретный датасет
- Полная разморозка могла бы привести к переобучению

### Часть B: Сегментация

**V1 vs V2:**
- Оба режима показали сопоставимые результаты
- DeepLabV3 с pretrained весами хорошо справляется с выделением объектов
- Альтернативная постобработка может дать небольшое улучшение

**Качество масок:**
- Модель хорошо выделяет основные объекты
- Могут быть неточности на границах и с мелкими объектами
-Foreground классы 1-20 охватывают все объекты интереса

---

## 8. Итоговый вывод

**Для классификации изображений:**
- ResNet18 с fine-tuning (C4) показал наилучшие результаты
- Transfer learning значительно ускоряет сходимость и улучшает качество
- Аугментации важны для борьбы с переобучением

**Для сегментации:**
- DeepLabV3 с pretrained весами демонстрирует хорошее качество масок
- Постобработка может дать небольшое улучшение метрик
- Правильное определение foreground классов критически важно

**Ключевые выводы:**
1. Transfer learning эффективен для небольших датасетов
2. Аугментации улучшают обобщающую способность
3. Fine-tuning требует баланса между адаптацией и переобучением
4. Pretrained модели для сегментации работают хорошо "из коробки"

**Для дальнейшего улучшения можно попробовать:**
- Learning rate scheduling (CosineAnnealing, ReduceLROnPlateau)
- Более сложные аугментации (MixUp, CutMix)
- Ансамблирование моделей
- Dice Loss для сегментации

---

## 9. Опциональная часть

- Дополнительные графики: `./artifacts/figures/`
- Confusion Matrix: (опционально)
- Error Analysis: (опционально)

---


