# Sales-Forecasting-with-XGBoost

Решение задачи регрессии (прогноз продаж `num_sold`) с использованием **XGBoost**.

---

## Задача
Предсказать количество проданных товаров на основе даты, страны, магазина и типа товара.

##  Особенности
-  Циклические признаки (sin/cos) для учёта сезонности
-  Полиномиальные признаки и взаимодействия
-  Поддержка CPU/GPU (автоопределение NVIDIA)
-  GridSearchCV для подбора гиперпараметров
-  Метрики: RMSE, MAE, R², WAPE, MAPE
-  Визуализация: корреляции, predictions vs truth, ошибки

---

##  Быстрый старт

```bash
git clone https://github.com/your-username/sales-forecasting-s5e1.git
cd sales-forecasting-s5e1
pip install -r requirements.txt
# Поместите train.csv в ./project/data/
python src/main.py
```

## Структура

project/

├── data/train.csv       # Данные (CC BY 4.0)

├── src/main.py          # Основной скрипт

├── requirements.txt     # Зависимости

└── README.md

## Лицензия

Данные:   Data source: [Kaggle Playground S5E1](https://www.kaggle.com/competitions/playground-series-s5e1), licensed under CC BY 4.0
