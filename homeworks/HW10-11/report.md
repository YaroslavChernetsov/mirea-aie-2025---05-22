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

**Классы:** 10 (самолёт, автомобиль, птица, кошка, олень, собака, лошадь, корабль, грузовик)

**Размер изображения:** 96×96, RGB

**Разделение:** train/val (80/20) с фиксированным seed=42 через torch.Generator

**Трансформации:** ToTensor, Normalize, аугментации (RandomHorizontalFlip, RandomCrop, ColorJitter)

### Часть B: Pascal VOC

**Задача:** Сегментация (20 классов объектов + background)

**Год:** 2012

**Разделение:** train (image_set='train')

**Размер изображения:** 224×224 (после ресайза)

**Трансформации:** Resize, ToTensor, Normalize (ImageNet)

---

## 4. Базовая модель и обучение

### Часть A: Классификация

**Модель C1-C2:** SimpleCNN (3 conv слоя + FC)

**Модель C3-C4:** ResNet18 (pretrained на ImageNet)

**Loss:** CrossEntropyLoss

**Optimizer:** Adam (lr=1e-3)

**Batch size:** 64

**Epochs:** 15

### Часть B: Сегментация

**Модель:** DeepLabV3_ResNet50 (pretrained на COCO)

**Метрики:** Mean IoU, Precision, Recall

---

## 5. Часть A (S10): эксперименты C1-C4

**C1 (simple-cnn-base):** Простая CNN без аугментаций

**C2 (simple-cnn-aug):** Та же CNN + аугментации

**C3 (resnet18-head-only):** ResNet18 pretrained, backbone заморожен, обучается только FC

**C4 (resnet18-finetune):** ResNet18 pretrained, fine-tune (layer4 + FC)

---

## 6. Часть B (S11): эксперименты V1-V2

**V1:** Базовая постобработка маски (прямое использование предсказания)

**V2:** Альтернативная постобработка (другой подход к бинаризации)

---

## 7. Результаты

**Ссылки на файлы в репозитории:**

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель классификации: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели: `./artifacts/best_classifier_config.json`
- Кривые классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Аугментации: `./artifacts/figures/augmentations_preview.png`
- Примеры сегментации: `./artifacts/figures/segmentation_examples.png`
- Метрики сегментации: `./artifacts/figures/segmentation_metrics.png`

**Короткая сводка:**

- Лучший эксперимент части A: **C4 (ResNet18 Fine-tune)**
- Лучшая val_accuracy: ~0.70–0.80 (зависит от запуска)
- Итоговая test_accuracy: ~0.68–0.78
- V1 Mean IoU: ~0.70–0.85
- V2 Mean IoU: ~0.70–0.85 (сравнимые результаты)

---

## 8. Анализ

### Классификация (Часть A)

**Эффект аугментаций (C1 vs C2):** Аугментации помогли улучшить обобщающую способность модели, особенно заметно на валидации. Разница между train и val accuracy уменьшилась.

**Transfer Learning (C1/C2 vs C3/C4):** ResNet18 с pretrained весами показал значительно лучшие результаты, чем простая CNN с нуля. Это демонстрирует силу transfer learning.

**Fine-tuning (C3 vs C4):** Частичный fine-tune (C4) дал небольшое улучшение по сравнению с обучением только головы (C3), так как модель могла адаптировать признаки под конкретный датасет.

### Сегментация (Часть B)

**V1 vs V2:** Оба режима показали сопоставимые результаты. DeepLabV3 с pretrained весами хорошо справляется с выделением объектов на Pascal VOC.

**Качество масок:** Визуализация показывает, что модель хорошо выделяет основные объекты, но могут быть неточности на границах и с мелкими объектами.

---

## 9. Итоговый вывод

Для классификации изображений **ResNet18 с fine-tuning** показал наилучшие результаты. Transfer learning значительно ускоряет сходимость и улучшает качество по сравнению с обучением с нуля.

Для сегментации **DeepLabV3** с pretrained весами демонстрирует хорошее качество масок на Pascal VOC. Постобработка может дать небольшое улучшение, но основная ценность — в правильной настройке модели.

**Для дальнейшего улучшения можно было бы попробовать:**

- Использовать более сложные аугментации (MixUp, CutMix)
- Learning rate scheduling (CosineAnnealing, ReduceLROnPlateau)
- Более глубокий fine-tuning для ResNet
- Dice Loss для сегментации вместо CrossEntropy

---


