# New York City Taxi Trip Duration
Подробный блокнот со всеми этапами построения модели и с пояснениями находится в файле "New York Taxi.ipynb"
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
* Выделение признаков - написание функций выделения новых признаков + удаления старых для упрощения работы с данными
* Обучение на новых данных - Кодирование плюс стандартизация, обучение линейной регрессией, проверка качества 
* Удаление выбросов - удаляются все аномалии (дни когда было запрещено движение транспорта, слишком длительные и слишком большие по расстоянию поездки и тд)
* Обучение на новых данных без выбросов - Линейная регрессия, подбор гиперпараметров, проверка качества
* Использование Случайного леса - Random Forest, проверка качества
* Обучение Градиентным бустингом - используется библиотека xgboost
* Обратные преобразования - так как предсказывали логарифмированную переменную, делаем обратное преобразование и смотрим на метрики качества

