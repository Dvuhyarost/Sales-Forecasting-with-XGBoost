import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import sys
import subprocess

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_PATH = './project/data/train.csv'


def select_mode():
    """Выбор режима работы CPU/GPU для XGBoost.

    Returns
    -------
    str
        'xgb_cpu' или 'xgb_gpu' - строка, указывающая выбранный режим
    """
    try:
        import xgboost as xgb
        has_xgb, version = True, xgb.__version__
    except ImportError:
        print("XGBoost не установлен. Установите: pip install xgboost")
        sys.exit(1)

    gpu_available = False
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True,
                                text=True, timeout=2)
        gpu_available = (result.returncode == 0 and
                         'NVIDIA-SMI' in result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    print(f"XGBoost {version} установен")
    print(f"GPU: {'доступен' if gpu_available else 'недоступен'}")

    if not gpu_available:
        print("\nИспользуется CPU режим")
        return 'xgb_cpu'

    choice = input("\n1. XGBoost на CPU\n2. XGBoost на GPU\n\n"
                   "Выберите вариант (1 или 2): ").strip()
    return 'xgb_gpu' if choice == '2' else 'xgb_cpu'


def load_and_prepare_data():
    """Загрузка, очистка данных и извлечение признаков из даты.

    Returns
    -------
    pandas.DataFrame
        Подготовленные данные с дополнительными признаками
    """
    data = pd.read_csv(TRAIN_PATH)
    print(f"Загружено данных: {data.shape}")

    # удаление строк с пропусками в целевом признаке
    data = data.dropna(subset=['num_sold'])
    # Преобразование целевой переменной к числовому типу
    data['num_sold'] = pd.to_numeric(data['num_sold'], errors='coerce')
    # Удаление строк, где преобразование не удалось
    data = data.dropna(subset=['num_sold'])
    # Преобразование к целочисленному типу
    data['num_sold'] = data['num_sold'].astype(int)

    # Извлечение признаков из даты
    data['date'] = pd.to_datetime(data['date'])  # Преобразование строки в datetime
    data['year'] = data['date'].dt.year  # Год
    data['month'] = data['date'].dt.month  # Месяц
    data['day'] = data['date'].dt.day  # День месяца
    data['weekday'] = data['date'].dt.weekday  # День недели (0=понедельник)
    data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)  # Флаг выходного дня
    data['day_of_year'] = data['date'].dt.dayofyear  # День года

    print(f"Очищено данных: {data.shape}")
    print("\nСтатистика целевой переменной:")
    print(f"  Минимум: {data['num_sold'].min()}, "
          f"Максимум: {data['num_sold'].max()}")
    print(f"  Среднее: {data['num_sold'].mean():.1f}, "
          f"Медиана: {data['num_sold'].median()}")

    return data


def calculate_numeric_correlations(data):
    """Рассчитать и отобразить корреляции числовых признаков.

    Parameters
    ----------
    data : pandas.DataFrame
        Данные с числовыми признаками
    """
    # Выбор только числовых колонок
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    # Расчет корреляции с целевой переменной
    corr_with_target = data[numeric_cols].corr()['num_sold']
    # Удаление корреляции с самой собой и сортировка по убыванию
    corr_with_target = corr_with_target.drop('num_sold').sort_values(
        ascending=False)

    print("\nКорреляция с целевой переменной (num_sold):")
    print(corr_with_target.to_string(float_format=lambda x: f"{x:.3f}"))

    # Создание тепловой карты корреляционной матрицы
    corr_matrix = data[numeric_cols].corr().values
    labels = data[numeric_cols].columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1,
                   aspect='auto')

    # Настройка осей
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Добавление значений корреляции в ячейки
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "black" if abs(corr_matrix[i, j]) < 0.5 else "white"
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center",
                    va="center", color=color, fontsize=8)

    ax.set_title('Матрица корреляций числовых признаков')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def create_engineered_features(X):
    """Создание дополнительных признаков на основе существующих.

    Parameters
    ----------
    X : pandas.DataFrame
        Исходные признаки

    Returns
    -------
    pandas.DataFrame
        Признаки с добавленными engineered features
    """
    X = X.copy()  # Создание копии, чтобы не изменять исходные данные

    # Полиномиальные признаки для месяца
    if 'month' in X.columns:
        X['month_sq'] = X['month'] ** 2  # Квадрат месяца
        X['month_cu'] = X['month'] ** 3  # Куб месяца

    # Взаимодействие признаков
    if 'year' in X.columns and 'month' in X.columns:
        X['year_month'] = X['year'] * X['month']

    # Циклические признаки для дня недели (синус и косинус)
    if 'weekday' in X.columns:
        weekday_normalized = X['weekday'] / 7  # Нормализация к [0,1]
        X['weekday_sin'] = np.sin(2 * np.pi * weekday_normalized)
        X['weekday_cos'] = np.cos(2 * np.pi * weekday_normalized)

    # Циклические признаки для дня года
    if 'day_of_year' in X.columns:
        day_of_year_normalized = X['day_of_year'] / 365.25
        X['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year_normalized)
        X['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year_normalized)

    return X


def prepare_features(data, use_engineered_features=True):
    """Подготовка признаков для моделирования.

    Parameters
    ----------
    data : pandas.DataFrame
        Подготовленные данные
    use_engineered_features : bool
        Флаг добавления engineered features

    Returns
    -------
    tuple
        (X, y) - признаки и целевая переменная
    """
    # Базовые признаки для модели
    base_features = ['year', 'month', 'day', 'weekday', 'is_weekend',
                     'day_of_year', 'country', 'store', 'product']

    # Проверка наличия всех признаков в данных
    available_features = [col for col in base_features if col in data.columns]
    X = data[available_features].copy()  # Копия признаков
    y = data['num_sold']  # Целевая переменная

    # Если не нужно использовать engineered features, возвращаем базовые
    if not use_engineered_features:
        return X, y

    # Создание дополнительных признаков
    X = create_engineered_features(X)

    # Вывод информации о добавленных признаках
    new_features = [col for col in X.columns if col not in available_features]
    if new_features:
        print("\nДобавленные признаки:")
        print("-" * 40)
        for feat in new_features:
            print(f"  {feat}")

    return X, y


def create_model_pipeline(mode, numeric_features, categorical_features,
                          xgb_params=None):
    """Создание пайплайна модели XGBoost.

    Parameters
    ----------
    mode : str
        Режим работы: 'xgb_cpu' или 'xgb_gpu'
    numeric_features : list
        Список числовых признаков
    categorical_features : list
        Список категориальных признаков
    xgb_params : dict, optional
        Параметры XGBoost

    Returns
    -------
    sklearn.pipeline.Pipeline
        Пайплайн для обучения модели
    """
    import xgboost as xgb  # Импорт внутри функции, чтобы не вызывать ошибку при отсутствии

    # Создание препроцессора для разных типов признаков
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),  # Стандартизация числовых признаков
        ('cat', OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False), categorical_features)  # One-hot кодирование категориальных
    ])

    # Параметры по умолчанию для XGBoost
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 100,  # Количество деревьев
            'max_depth': 6,  # Максимальная глубина дерева
            'learning_rate': 0.1,  # Скорость обучения
            'random_state': RANDOM_STATE,  # Для воспроизводимости
            'objective': 'reg:squarederror'  # Функция потерь для регрессии
        }
    if mode == 'xgb_gpu':
        xgb_params['device'] = 'cuda'
    else:
        xgb_params['n_jobs'] = -1

    model = xgb.XGBRegressor(**xgb_params)  # Создание модели регрессии

    # Создание пайплайна: препроцессор + модель
    return Pipeline([('preprocessor', preprocessor), ('regressor', model)])


def calculate_regression_metrics(y_true, y_pred):
    """Расчет метрик регрессии.

    Parameters
    ----------
    y_true : array-like
        Истинные значения
    y_pred : array-like
        Предсказанные значения

    Returns
    -------
    dict
        Словарь с метриками регрессии
    """
    y_true = np.asarray(y_true)  # Преобразование к numpy массиву
    y_pred = np.asarray(y_pred)

    # Базовые метрики регрессии
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),  # Среднеквадратичная ошибка
        'mae': mean_absolute_error(y_true, y_pred),  # Средняя абсолютная ошибка
        'r2': r2_score(y_true, y_pred)  # Коэффициент детерминации R²
    }

    # Расчет WAPE (Weighted Absolute Percentage Error)
    abs_errors = np.abs(y_true - y_pred)  # Абсолютные ошибки
    eps = 1e-10  # Малое число для избежания деления на ноль
    sum_abs_y = np.sum(np.abs(y_true))

    metrics['wape'] = (np.sum(abs_errors) / (sum_abs_y + eps)) * 100

    # Расчет MAPE (Mean Absolute Percentage Error) и доли ошибок >10%
    mask_nonzero = np.abs(y_true) > eps  # Маска для ненулевых значений
    if np.sum(mask_nonzero) > 0:
        mape_vals = abs_errors[mask_nonzero] / np.abs(y_true[mask_nonzero])
        metrics['mape'] = np.mean(mape_vals) * 100  # Средняя абсолютная процентная ошибка
        metrics['error_gt_10pct'] = np.mean(mape_vals > 0.10) * 100  # Процент ошибок >10%
    else:
        metrics['mape'] = 0.0
        metrics['error_gt_10pct'] = 0.0

    return metrics


def evaluate_model_performance(pipeline, X_test, y_test, model_name):
    """Оценка производительности модели.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Обученная модель
    X_test : pandas.DataFrame
        Тестовые признаки
    y_test : pandas.Series
        Тестовые значения
    model_name : str
        Название модели для вывода

    Returns
    -------
    dict
        Результаты оценки модели
    """
    y_pred = pipeline.predict(X_test)  # Получение предсказаний
    metrics = calculate_regression_metrics(y_test, y_pred)  # Расчет метрик

    # Вывод результатов
    print(f"\n{model_name}:")
    print("-" * 50)
    print(f"  RMSE:    {metrics['rmse']:.2f}")
    print(f"  MAE:     {metrics['mae']:.2f}")
    print(f"  R²:      {metrics['r2']:.4f}")
    print(f"  WAPE:    {metrics['wape']:.2f}%")
    print(f"  MAPE:    {metrics['mape']:.2f}%")
    print(f"  Ошибки >10%: {metrics['error_gt_10pct']:.1f}%")

    # Добавление предсказаний и истинных значений для последующей визуализации
    metrics.update({'predictions': y_pred, 'true_values': y_test.values})
    return metrics


def tune_hyperparameters(pipeline, X_train, y_train, mode):
    """Настройка гиперпараметров модели.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Базовая модель
    X_train : pandas.DataFrame
        Обучающие признаки
    y_train : pandas.Series
        Обучающие значения
    mode : str
        Режим работы ('xgb_cpu' или 'xgb_gpu')

    Returns
    -------
    sklearn.pipeline.Pipeline
        Настроенная модель
    """
    # Сетка параметров для поиска
    param_grid = {
        'regressor__n_estimators': [50, 100, 150],  # Количество деревьев
        'regressor__max_depth': [3, 6, 9],  # Глубина деревьев
        'regressor__learning_rate': [0.01, 0.1, 0.3]  # Скорость обучения
    }

    # Настройка параллельных вычислений
    n_jobs = -1 if mode == 'xgb_cpu' else 1  # Для GPU не используем параллелизм

    # Поиск по сетке с кросс-валидацией
    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=3,  # 3-х кратная кросс-валидация
        scoring='neg_mean_squared_error',  # Метрика для оптимизации
        n_jobs=n_jobs,
        verbose=0  # Без вывода прогресса
    )

    grid_search.fit(X_train, y_train)  # Обучение с поиском лучших параметров

    print("\nЛучшие параметры:")
    for param, value in grid_search.best_params_.items():
        param_name = param.split('__')[1]  # Извлечение имени параметра
        print(f"  {param_name}: {value}")

    return grid_search.best_estimator_  # Возврат лучшей модели


def create_subplot_metrics_comparison(ax, results_basic, results_tuned):
    """Создание графика сравнения метрик.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Ось для отрисовки
    results_basic : dict
        Результаты базовой модели
    results_tuned : dict
        Результаты настроенной модели
    """
    metrics = ['RMSE', 'MAE', 'R²', 'WAPE(%)']  # Список метрик для сравнения
    basic_vals = [results_basic['rmse'], results_basic['mae'],
                  results_basic['r2'], results_basic['wape']]
    tuned_vals = [results_tuned['rmse'], results_tuned['mae'],
                  results_tuned['r2'], results_tuned['wape']]

    x = np.arange(len(metrics))  # Позиции по оси X
    width = 0.35  # Ширина столбцов

    # Столбчатая диаграмма для базовой модели
    ax.bar(x - width/2, basic_vals, width, label='Базовая', alpha=0.85,
           color='#4c72b0', edgecolor='black', linewidth=0.5)
    # Столбчатая диаграмма для настроенной модели
    ax.bar(x + width/2, tuned_vals, width, label='Настроенная',
           alpha=0.85, color='#55a868', edgecolor='black', linewidth=0.5)

    # Настройка осей и легенды
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=10)
    ax.set_title('Сравнение основных метрик', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.4, axis='y')  # Сетка только по оси Y

    # Добавление значений над столбцами
    y_offset = max(max(basic_vals), max(tuned_vals)) * 0.03
    for i, (basic, tuned) in enumerate(zip(basic_vals, tuned_vals)):
        fmt = '.1f' if i != 2 else '.4f'  # Разный формат для R²
        ax.text(i - width/2, basic + y_offset, f'{basic:{fmt}}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2c4c70')
        ax.text(i + width/2, tuned + y_offset, f'{tuned:{fmt}}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2c5a3a')

    # Установка пределов оси Y
    y_max = max(max(basic_vals), max(tuned_vals)) * 1.12
    ax.set_ylim(0, y_max)


def create_subplot_predictions_vs_truth(ax, results_tuned):
    """Создание графика предсказаний против истинных значений.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Ось для отрисовки
    results_tuned : dict
        Результаты настроенной модели
    """
    y_true = results_tuned['true_values']
    y_pred = results_tuned['predictions']
    max_val = max(y_true.max(), y_pred.max()) * 1.05  # Максимальное значение с запасом

    # Гексагональный бинаринг для визуализации плотности точек
    hb = ax.hexbin(y_true, y_pred, gridsize=40, cmap='Blues',
                   mincnt=1, linewidth=0.1)
    # Идеальная линия (диагональ)
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2,
            label='Идеальная линия')
    ax.set_xlabel('Истинные значения', fontsize=10)
    ax.set_ylabel('Предсказанные значения', fontsize=10)
    ax.set_title('Предсказания vs Истина', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return hb  # Возврат объекта для цветовой шкалы


def create_subplot_error_distribution(ax, results_tuned):
    """Создание графика распределения ошибок.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Ось для отрисовки
    results_tuned : dict
        Результаты настроенной модели
    """
    y_true = results_tuned['true_values']
    y_pred = results_tuned['predictions']
    errors = y_true - y_pred  # Расчет ошибок

    # Гистограмма распределения ошибок
    n, bins, patches = ax.hist(errors, bins=40, alpha=0.75,
                               color='#c44e52', edgecolor='black',
                               linewidth=0.5)
    # Вертикальная линия нулевой ошибки
    ax.axvline(0, color='black', linestyle='--', linewidth=2,
               label='Нулевая ошибка')

    # Линия средней ошибки
    mean_err = np.mean(errors)
    ax.axvline(mean_err, color='navy', linestyle='-', linewidth=2,
               label=f'Средняя ошибка: {mean_err:.2f}')

    ax.set_xlabel('Ошибка (истина − предсказание)', fontsize=10)
    ax.set_ylabel('Частота', fontsize=10)
    ax.set_title('Распределение ошибок', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def create_subplot_relative_metrics(ax, results_basic, results_tuned):
    """Создание графика относительных метрик.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Ось для отрисовки
    results_basic : dict
        Результаты базовой модели
    results_tuned : dict
        Результаты настроенной модели
    """
    rel_metrics = ['MAPE(%)', 'Ошибки >10%']  # Относительные метрики
    basic_rel = [results_basic['mape'], results_basic['error_gt_10pct']]
    tuned_rel = [results_tuned['mape'], results_tuned['error_gt_10pct']]

    x_rel = np.arange(len(rel_metrics))
    width = 0.35

    # Столбчатые диаграммы для относительных метрик
    ax.bar(x_rel - width/2, basic_rel, width, label='Базовая',
           alpha=0.85, color='#4c72b0', edgecolor='black')
    ax.bar(x_rel + width/2, tuned_rel, width, label='Настроенная',
           alpha=0.85, color='#55a868', edgecolor='black')

    # Настройка осей
    ax.set_xticks(x_rel)
    ax.set_xticklabels(rel_metrics, rotation=45, ha='right', fontsize=10)
    ax.set_title('Относительные метрики', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='y')

    # Добавление значений над столбцами
    offset = max(max(basic_rel), max(tuned_rel)) * 0.04
    for i, (b, t) in enumerate(zip(basic_rel, tuned_rel)):
        ax.text(i - width/2, b + offset, f'{b:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2c4c70')
        ax.text(i + width/2, t + offset, f'{t:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2c5a3a')

    # Установка пределов оси Y
    y_max = max(max(basic_rel), max(tuned_rel)) * 1.15
    ax.set_ylim(0, y_max)


def visualize_results(results_basic, results_tuned, model_name):
    """Визуализация результатов сравнения моделей.

    Parameters
    ----------
    results_basic : dict
        Результаты базовой модели
    results_tuned : dict
        Результаты настроенной модели
    model_name : str
        Название модели
    """
    # Создание сетки графиков 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Заполнение каждого субграфика соответствующей функцией
    create_subplot_metrics_comparison(axes[0, 0], results_basic, results_tuned)
    hb = create_subplot_predictions_vs_truth(axes[0, 1], results_tuned)
    create_subplot_error_distribution(axes[1, 0], results_tuned)
    create_subplot_relative_metrics(axes[1, 1], results_basic, results_tuned)

    # Добавление цветовой шкалы для гексагонального графика
    fig.colorbar(hb, ax=axes[0, 1], shrink=0.8, label='Плотность точек')

    # Настройка компоновки
    plt.tight_layout(pad=2.5, h_pad=2.0, w_pad=1.5)
    plt.suptitle(f'Результаты: {model_name}', fontsize=14,
                 fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    plt.show()


def print_results_comparison(results_basic, results_tuned, model_name):
    """Вывод сравнения результатов моделей.

    Parameters
    ----------
    results_basic : dict
        Результаты базовой модели
    results_tuned : dict
        Результаты настроенной модели
    model_name : str
        Название модели
    """
    # Создание DataFrame для удобного сравнения
    comparison = pd.DataFrame({
        'Модель': ['Базовая (без engineered фич)',
                   'Настроенная (с engineered фич + GridSearch)'],
        'RMSE': [results_basic['rmse'], results_tuned['rmse']],
        'MAE': [results_basic['mae'], results_tuned['mae']],
        'R²': [results_basic['r2'], results_tuned['r2']],
        'WAPE(%)': [results_basic['wape'], results_tuned['wape']],
        'MAPE(%)': [results_basic['mape'], results_tuned['mape']],
        'Ошибки >10%': [f"{results_basic['error_gt_10pct']:.1f}%",
                        f"{results_tuned['error_gt_10pct']:.1f}%"]
    })

    print("\nСРАВНЕНИЕ БАЗОВОЙ И НАСТРОЕННОЙ МОДЕЛЕЙ:")
    print("=" * 100)
    print(comparison.to_string(index=False,
          float_format=lambda x: f'{x:.3f}' if isinstance(x, float) else x))

    def calc_improvement(basic, tuned, higher_is_better=False):
        """Расчет процентного улучшения.

        Parameters
        ----------
        basic : float
            Значение базовой модели
        tuned : float
            Значение настроенной модели
        higher_is_better : bool
            True если большее значение лучше

        Returns
        -------
        float
            Процент улучшения
        """
        if np.isclose(basic, 0):  # Избежание деления на ноль
            return 0.0
        if higher_is_better:
            return (tuned - basic) / abs(basic) * 100  # Для R² (чем больше, тем лучше)
        return (basic - tuned) / basic * 100  # Для ошибок (чем меньше, тем лучше)

    print("\nУЛУЧШЕНИЯ:")
    print("-" * 60)

    # Список метрик для расчета улучшения
    metrics = [
        ('RMSE', 'rmse', False),
        ('MAE', 'mae', False),
        ('R²', 'r2', True),
        ('WAPE', 'wape', False),
        ('MAPE', 'mape', False),
        ('Ошибки >10%', 'error_gt_10pct', False)
    ]

    # Расчет и вывод улучшений для каждой метрики
    for display_name, metric_name, higher_is_better in metrics:
        improvement = calc_improvement(results_basic[metric_name],
                                       results_tuned[metric_name],
                                       higher_is_better)
        print(f"  {display_name:15} {improvement:+.1f}%")

    # Расчет улучшений для ключевых метрик
    r2_improvement = calc_improvement(results_basic['r2'],
                                      results_tuned['r2'], True)
    mae_improvement = calc_improvement(results_basic['mae'],
                                       results_tuned['mae'], False)

    print("\nВЫВОДЫ:")
    print("-" * 60)
    msg1 = f"• Модель с engineered признаками объясняет "
    msg2 = f"на {r2_improvement:.1f}% больше дисперсии."
    print(msg1 + msg2)
    msg3 = f"• Средняя абсолютная ошибка снизилась "
    msg4 = f"на {mae_improvement:.1f}%."
    print(msg3 + msg4)


def main():
    """Основная функция выполнения анализа."""
    print("=" * 80)
    print("РЕГРЕССИОННЫЙ АНАЛИЗ ПРОДАЖ")
    print("=" * 80)

    mode = select_mode()
    model_name = 'XGBoost (GPU)' if mode == 'xgb_gpu' else 'XGBoost (CPU)'

    print("\n1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("-" * 60)

    data = load_and_prepare_data()

    print("\n2. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    print("-" * 60)
    calculate_numeric_correlations(data)

    print("\n3. ПОДГОТОВКА ПРИЗНАКОВ")
    print("-" * 60)

    # Подготовка признаков в двух вариантах
    X_basic, y = prepare_features(data, use_engineered_features=False)
    X_enhanced, _ = prepare_features(data, use_engineered_features=True)

    # Разделение данных на обучающую и тестовую выборки
    X_train_basic, X_test_basic, y_train, y_test = train_test_split(
        X_basic, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train_enh, X_test_enh, _, _ = train_test_split(
        X_enhanced, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\nРазмеры выборок:")
    print(f"  Обучающая: {X_train_basic.shape}, "
          f"Тестовая: {X_test_basic.shape}")

    print("\n4. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
    print("-" * 60)

    print("\nБазовая модель (без engineered features):")
    numeric_basic = X_basic.select_dtypes(include=[np.number]).columns.tolist()
    categorical_basic = X_basic.select_dtypes(
        include=['object']).columns.tolist()

    # Создание, обучение и оценка базовой модели
    pipeline_basic = create_model_pipeline(mode, numeric_basic,
                                           categorical_basic)
    pipeline_basic.fit(X_train_basic, y_train)
    results_basic = evaluate_model_performance(
        pipeline_basic, X_test_basic, y_test, f"{model_name} (базовая)"
    )

    print("\n\nМодель с engineered features (без настройки):")
    numeric_enh = X_enhanced.select_dtypes(
        include=[np.number]).columns.tolist()
    categorical_enh = X_enhanced.select_dtypes(
        include=['object']).columns.tolist()

    # Создание, обучение и оценка модели с дополнительными признаками
    pipeline_enh = create_model_pipeline(mode, numeric_enh, categorical_enh)
    pipeline_enh.fit(X_train_enh, y_train)
    results_enh = evaluate_model_performance(
        pipeline_enh, X_test_enh, y_test,
        f"{model_name} (с engineered features)"
    )

    print("\n\n5. НАСТРОЙКА ГИПЕРПАРАМЕТРОВ")
    print("-" * 60)

    # Настройка гиперпараметров для улучшенной модели
    pipeline_tuned = tune_hyperparameters(pipeline_enh, X_train_enh,
                                          y_train, mode)
    results_tuned = evaluate_model_performance(
        pipeline_tuned, X_test_enh, y_test, f"{model_name} (настроенная)"
    )

    print("\n6. АНАЛИЗ И ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("-" * 60)

    # Визуализация и вывод результатов
    visualize_results(results_basic, results_tuned, model_name)
    print_results_comparison(results_basic, results_tuned, model_name)


if __name__ == "__main__":
    main() 