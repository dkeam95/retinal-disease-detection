# Отчет по результатам анализа проекта Retinal Disease Detection

В данном отчете представлены результаты статического анализа, исследования кодовой базы и выполнения тестов для проекта распознавания ретинальных патологий ( Diabetic Retinopathy severity grading).

---

## 1. Критические ошибки и баги (High Priority)

### 1.1. Отсутствующие модули и сломанные импорты в `main.py`
В [`main.py`](file:///D:/python_projects/retinal-disease-detection/main.py) задекларированы режимы работы, которые импортируют несуществующие скрипты из папки `scripts/`:
* Режим `benchmark-xai` пытается выполнить: `from scripts.evaluation.benchmark_xai import main as run_xai`. Файл [`benchmark_xai.py`](file:///D:/python_projects/retinal-disease-detection/scripts/evaluation/benchmark_xai.py) отсутствует.
* Режим `benchmark-det` пытается выполнить: `from scripts.evaluation.compare_detection_rounds import main as run_det_bench`. Файл [`compare_detection_rounds.py`](file:///D:/python_projects/retinal-disease-detection/scripts/evaluation/compare_detection_rounds.py) отсутствует.
* Режим `train-detection` пытается выполнить: `from scripts.training.train_detection import main as run_train_det`. Директория `scripts/training` отсутствует.
* **Следствие:** Запуск `main.py` в указанных режимах немедленно завершается с ошибкой `ImportError`.

### 1.2. Ошибка NameError (отсутствие глобального импорта `torch`) в CLI-модулях
В файлах [`src/evaluation/eval.py`](file:///D:/python_projects/retinal-disease-detection/src/evaluation/eval.py#L63) и [`src/inference/predict.py`](file:///D:/python_projects/retinal-disease-detection/src/inference/predict.py#L60) используется вызов `torch.cuda.is_available()`. Однако импорт `import torch` находится внутри блока `if __name__ == "__main__":`.
* **Следствие:** При импорте функций или запуске методов из сторонних скриптов (не как `__main__`) возникает ошибка `NameError: name 'torch' is not defined`.

### 1.3. Ошибка AttributeError (несовместимость с `BatchSample`) в `ModelEvaluator.evaluate`
В [`src/evaluation/evaluator.py:L124`](file:///D:/python_projects/retinal-disease-detection/src/evaluation/evaluator.py#L124) используется строчка:
```python
images, targets = batch.images, batch.labels
```
Однако реальный DataLoader возвращает батчи в формате класса [`BatchSample`](file:///D:/python_projects/retinal-disease-detection/src/dataloader/factory.py#L24), который имеет атрибуты в единственном числе: `image` и `label`.
* **Следствие:** При оценке модели на реальных данных выполнение падает с ошибкой `AttributeError: 'BatchSample' object has no attribute 'images'`. В тестах эта ошибка была скрыта, так как использовался `TensorDataset`, возвращающий кортежи (списки), что уводило выполнение в другую ветку `isinstance(batch, (list, tuple))`.

### 1.4. Ошибка AttributeError (пропущенный аргумент `--config`) в `src/inference/predict.py`
В [`src/inference/predict.py:L57`](file:///D:/python_projects/retinal-disease-detection/src/inference/predict.py#L57) происходит загрузка конфигурации: `config = ConfigLoader.load(args.config)`. Однако в `argparse.ArgumentParser` этого скрипта аргумент `--config` вообще не добавлен.
* **Следствие:** Запуск CLI-скрипта падает с ошибкой `AttributeError: 'Namespace' object has no attribute 'config'`.

### 1.5. Падение при инициализации GUI tkinter в headless-окружениях
Скрипт [`test_cls.py:L61`](file:///D:/python_projects/retinal-disease-detection/test_cls.py#L61) вызывает `root = tk.Tk()` для открытия диалогового окна выбора файлов.
* **Следствие:** При запуске в Docker, на сервере без дисплея или через SSH-сессию код аварийно завершится с ошибкой Tcl/Tk (`_tkinter.TclError: no display name...`), даже если пользователь передал аргумент `--image` через консоль, так как инициализация GUI происходит до проверки пути.

### 1.6. Несовместимость `torch.compile` с `CheckpointManager` при инференсе
При обучении на GPU модель оборачивается в `torch.compile()` (класс `OptimizedModule`). При сохранении состояния через `CheckpointManager` ключи в `state_dict()` получают префикс `_orig_mod.`.
* **Следствие:** При попытке загрузить такой чекпоинт в стандартную модель для инференса ([`src/inference/predictor.py:L115`](file:///D:/python_projects/retinal-disease-detection/src/inference/predictor.py#L115)) или в тестах без CUDA, PyTorch выбрасывает `RuntimeError` из-за несовпадения ключей.

---

## 2. Архитектурные узкие места (Bottlenecks) и проблемы производительности

### 2.1. Критическое замедление при инициализации `WeightedRandomSampler`
Для борьбы с дисбалансом классов в [`src/dataloader/sampler.py:L39`](file:///D:/python_projects/retinal-disease-detection/src/dataloader/sampler.py#L39) собираются метки классов всего датасета:
```python
labels = [dataset[index].label for index in range(len(dataset))]
```
Обращение `dataset[index]` в [`RetinalDataset`](file:///D:/python_projects/retinal-disease-detection/src/dataset/dataset.py#L130) считывает изображение с диска и декодирует его через `cv2.imread`.
* **Следствие:** При запуске обучения с весовым сэмплером происходит синхронное чтение и декодирование **всех** изображений датасета на CPU. Для больших датасетов это приводит к зависанию на десятки минут перед стартом первой эпохи. Необходимо считывать метки из кэшированных аннотаций (`dataset._annotations`).

### 2.2. Жестко заданный размер изображения в `collate_datasamples` игнорирует конфигурацию
Функция [`collate_datasamples`](file:///D:/python_projects/retinal-disease-detection/src/dataloader/factory.py#L34) принимает аргумент `target_size` с дефолтным значением `(224, 224)`. При инициализации PyTorch `DataLoader` передается `collate_fn=collate_datasamples` без частичного применения (например, через `functools.partial`).
* **Следствие:** Все батчи при обучении ресайзятся к разрешению **224x224**, даже если в конфигурации (например, в [`exp_07_convnext_tiny_v4.yaml`](file:///D:/python_projects/retinal-disease-detection/configs/experiments/exp_07_convnext_tiny_v4.yaml#L9)) явно указан размер **512x512**. Это приводит к падению качества модели и несовместимости с инференсом, который использует корректный размер из конфига.

---

## 3. Несоответствия стандартам кода / архитектуре проекта

### 3.1. Дублирование фабрик моделей (TIMM vs Torchvision) и неиспользуемый домен `src/model/`
* **Спецификация:** Согласно [`docs/DESIGN_DOC.md`](file:///D:/python_projects/retinal-disease-detection/docs/DESIGN_DOC.md#L31) и принципам архитектуры, проект должен использовать TIMM backbones с отключенной головой (`num_classes=0`), а затем оборачивать их в кастомные классификаторы. Эта логика реализована в папке [`src/model/`](file:///D:/python_projects/retinal-disease-detection/src/model).
* **Фактическая реализация:** Все основные модули (`train.py`, `predictor.py`, `eval.py`) импортируют и используют [`src/training/model_factory.py`](file:///D:/python_projects/retinal-disease-detection/src/training/model_factory.py), которая построена на моделях из библиотеки `torchvision.models` и производит замену классификатора вручную.
* **Следствие:** Весь домен `src/model/` является «мертвым кодом» и нигде не задействован в проде. Это нарушает принципы SRP и чистоты архитектуры.

### 3.2. Полный обход пайплайна аугментации Albumentations при обучении
* **Спецификация:** Заявлена поддержка гибкого пайплайна аугментаций (flips, rotations, contrast) через Albumentations в [`src/preprocessing/pipeline.py`](file:///D:/python_projects/retinal-disease-detection/src/preprocessing/pipeline.py).
* **Фактическая реализация:** Пайплайны аугментации не применяются в [`src/trainer/train.py`](file:///D:/python_projects/retinal-disease-detection/src/trainer/train.py) и не передаются в датасет. Вместо этого в `collate_datasamples` реализована ручная конвертация и нормализация средствами OpenCV и PyTorch.
* **Следствие:** Обучение моделей идет **без аугментации данных**, что приводит к быстрому переобучению.

### 3.3. Нарушение PEP 8 в `ModelEvaluator`
В методе [`ModelEvaluator.from_checkpoint`](file:///D:/python_projects/retinal-disease-detection/src/evaluation/evaluator.py#L58) первым аргументом объявлен `self` вместо общепринятого `cls` для `@classmethod`.

### 3.4. Ошибки линтера Ruff (750+ предупреждений)
Запуск `ruff check .` выявил 754 предупреждения:
* Большинство ошибок связано с превышением длины строки в комментариях и тестах (правило `E501`, лимит 88 символов).
* Наличие вызовов `zip()` без явного указания параметра `strict=` (правило `B905`) в файлах тестов (например, в [`tests/trainer/test_step.py`](file:///D:/python_projects/retinal-disease-detection/tests/trainer/test_step.py#L87)).
