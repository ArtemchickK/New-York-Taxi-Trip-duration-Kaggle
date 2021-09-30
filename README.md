# New York City Taxi Trip Duration
## Задача
Соревнование на Kaggle - https://www.kaggle.com/c/nyc-taxi-trip-duration/overview. Нужно предсказать продолжительность поездки такси по следующим признакам: 
* id - идентификатор поездки
* vendor_id - код провайдера, от которого пришла информация о поездке
* pickup_datetime - время старта поездки
* dropoff_datetime - время окончания поездки
* passenger_count - число пассажиров (вводится водителем)
* pickup_longitude - долгота точки посадки
* pickup_latitude - широта точки посадки
* dropoff_longitude - долгота точки высадки
* dropoff_latitude - долгота точки высадки
* store_and_fwd_flag - равно Y, если информация о поездке какое-то время хранилась в памяти таксометра из-за отсутствия связи; иначе принимает значение N
* trip_duration - продолжительность поездки в секундах
## Результаты
Наилучший результат был достигнут с помощью метода градиентного бустинга. Удалось уменьшить ошибку в среднем до 3.6 минут на метрике MAE и до 5.9 минут на метрике RMSE. При этом константные предсказания (целевую переменную оценили средним значением) до обработки данных и выделения признаков были следующими: MAE = 9.5 минут, а RMSE = 87.5 минут (из-за большого количества выбросов).
## Решение
### EDA (Разведовочный анализ)
* Подготовка данных - проверка тренировочных и тестовых данных, проверка пропущенных значений, проверка уникальности, преобразование типов
* Проверка распределения целевой переменной - логарифмирование целевой переменной, поиск выбросов, карта поездок
* Работа с расстоянием - подсчет расстояния по координатам, поиск выбросов, проверка корреляции расстояния с длительностью поездки
* Работа с datetime переменной - выделение признаков (число месяца, день недели, час дня и тд), исследование их распределений, поиск выбросов
### Обучение на первоначальных данных и константные предсказания
* Обработка переменных и обучение - One-hot и StandardScaler + LinearRegressor
* Константные предсказания - оценка целевой переменной средним значением
### Обучение на новых признаках
* Выделение признаков
* Очистка старых признаков
* Удаление выбросов
* Обучение градиентным бустингом

## Оглавление:
*  Разведовочный анализ
    * Подготовка данных
    * Целевая переменная и координаты
    * Работа с переменной расстояния
    * Работа с pickup_datetime переменной   
* Обучение на необработанных данных и константные предположения
* Обучение на новых признаках
    * Функции предобработки и выделения признаков
    * Обучение на новых признаках с выбросами
    * Удаление выбросов
    * Обучение на данных без выбросов и с новыми признаками
    * Подбор гиперпараметров
    * Random Forest
    * Градиентный бустинг
    * Обратные преобразования
