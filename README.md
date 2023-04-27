# Skillbox_Diploma_ML_Prediction
Разработка и обучение модели, предсказывающей совершение клиентами целевых действий на сайте 'СберАвтоподписка'
Проект в качестве итогового задания по курсу 'Введение в Data Science' от Skillbox.

Цель - разработать приложение, которое будет предсказывать совершение пользователем одного из целевых действий - 'Оставить заявку' или 'Заказать звонок' по времени визита 'visit_*', рекламным меткам 'utm_*', характеристикам устройства 'device_*' и местоположению 'geo_*'. Это должен быть сервис, который будет брать на вход все атрибуты, типа utm_, device_, geo_*, и отдавать на выход 0/1 (1 — если пользователь совершит любое целевое действие).

Целевые метрики: roc-auc > 0.65, время предсказания не более 3 секунд.

# Описание структуры проекта:


	step_1_data_analises.ipynb - анализ и подготовка датасет для моделирования.
	step_2_model_research.ipynb- моделирование, выбор лучшей модели.
	
	
# ШАГ1 - ЭТАП АНАЛИЗА ДАТАСЕТ hits и sessions

## df_hits Пропуски, дубликаты, 0-значения обработка 
Есть пропуски (event_value, hit_time, hit_referer) > 58% удаляются.
Есть пропуски (event_label) < 40% - будут заполняться самым встречающимся значение.

![image](https://user-images.githubusercontent.com/104129537/234815097-ea6b61be-05b1-42c1-90bf-1b809ab7d121.png)

## Колонку даты преобразуем в тип-дата и создадим дополнительные признаки (месяц - visit_date_month, день недели - Visit_date_weekday)
)
![image](https://user-images.githubusercontent.com/104129537/234818952-e2f3ebe3-c202-4920-a73c-0f2ba1ea93cc.png)


## Из колонки Hit_page_path извлечем данные о модели и марке авто,  в дальнейшем будет использована в OHE
в топ авто такие марки как volkswagen, mercedes-benz, nissan, skoda, bmw, lada-vaz, kia
![image](https://user-images.githubusercontent.com/104129537/234831053-5699bd59-58fc-43b4-b6d7-ae70e69dd403.png)


## Из колонки Event_action (целевая колонка, из которой необходимо сделать предсказание по целевым действиям) создадим target, куда внесем 8 целевых действий, данными в условиях задачи.   

![image](https://user-images.githubusercontent.com/104129537/234832130-ae4b384b-97c4-428d-bd67-2d0cc3445787.png)

	Всего целевых действий = 74140 из 12462536 что составляет  0.59 %

	sub_car_claim_click = 36364 === 0.29 %
	sub_submit_success = 14484 === 0.12 %
	sub_car_claim_submit_click = 11931 === 0.1 %
	sub_open_dialog_click = 6831 === 0.05 %
	sub_car_request_submit_click = 1975 === 0.02 %
	sub_call_number_click = 1372 === 0.01 %
	sub_callback_submit_click = 1174 === 0.01 %
	sub_custom_question_submit_click = 9 === 0.0 %
	
	## Df_sessions Пропуски, дубликаты, 0-значения обработка 
Есть пропуски (device_model, utm_keyword, device_os) > 58% удаляются.
Есть пропуски (device_brand, utm_adcontent, utm_campaign, utm_source) < 40% - будут заполняться самым встречающимся значение.

![image](https://user-images.githubusercontent.com/104129537/234862693-205016ab-d1cf-4a45-99e3-8bf36fc5b525.png)

	## Процент целевых действий в колонке target
![image](https://user-images.githubusercontent.com/104129537/234866027-3f6f4030-4309-48b2-8174-9111c6ca92bc.png)


## Колонку даты преобразуем в тип-дата и создадим дополнительные признаки (выходной день - Visit_date_weekend)
![image](https://user-images.githubusercontent.com/104129537/234835268-21bea7e8-efac-4782-a835-d32803a5eafa.png)


## Колонку время преобразуем в тип-время и создадим дополнительные признаки (посещение по часам - Visit_time_hour, посещение по часам в ночное время Visit_time_night)

![image](https://user-images.githubusercontent.com/104129537/234863090-2277d094-089c-41ab-aa16-923e1c7cb60c.png)


## Колонка Device_category имеет 3 параметра в дальнейшем будет использована в OHE
![image](https://user-images.githubusercontent.com/104129537/234863729-87b6ea0f-87da-4e47-9dde-ef6481b0544c.png)



## Из колонки Device_screen_resolution создадим новые признаки (device_screen_width, device_screen_height, device_screen_area, device_screen_ratio)

![image](https://user-images.githubusercontent.com/104129537/234864105-29d4293f-feb8-426f-badb-edff98f8453e.png)
![image](https://user-images.githubusercontent.com/104129537/234864147-3638a1f6-5446-4123-b7b6-f13cb57465cd.png)



## Из колонки Geo_city  cделаем колонку больших городов (big_cities)
big_cities = ['Moscow', 'Saint Petersburg', 'Novosibirsk', 'Yekaterinburg', 
              'Kazan', 'Nizhny Novgorod', 'Chelyabinsk', 'Samara', 'Omsk', 
              'Rostov-on-Don', 'Ufa', 'Krasnoyarsk', 'Voronezh', 'Perm', 
              'Volgograd', 'Krasnodar', 'Saratov', 'Tyumen']

![image](https://user-images.githubusercontent.com/104129537/234864237-063e07a2-770b-4b12-b582-608aac908cc6.png)


## Из колонки Geo_city  cделаем колонку расстояние до Москвы (geo_city_distance_from_moscow)

![image](https://user-images.githubusercontent.com/104129537/234864323-2371e332-8944-4860-bc60-8de4f4c0be39.png)
![image](https://user-images.githubusercontent.com/104129537/234864514-f5e5e0e9-48f7-47aa-8d36-9709c35fdf12.png)


# Поиск корреляций
![image](https://user-images.githubusercontent.com/104129537/234864642-3cf1f464-dadb-424c-96b9-7b9f9b6d6fcd.png)


# ВЫВОДЫ по 1 этапу - очистка и подготовка данных для моделирования. 
Сервис "СберАвтоподписка" - новый продукт, потому, клинты его тщательно изучают. Только 7.68 % сессий завершилось целевым действием.
Наблюдения:
В дневное время и в первой половине недели больше всего производится целевых действий.
Чем больше раз пользователь посещает сайт, тем скорее он совершит целевое действие.
В топах по выбору авто стоят volkswagen, mercedes-benz, nissan, skoda, bmw, lada-vaz, kia

# ШАГ2 - ЭТАП МОДЕЛИРОВАНИЯ, ПОДБОРА ЛУЧШЕЙ МОДЕЛИ, ОБОСНОВАНИЕ

## Нормализация данных
'device_screen_width', 'device_screen_height', 'device_screen_area', 'device_screen_ratio', 'geo_city_distance_from_moscow', 'device_screen_width','device_screen_height', 'device_screen_area', 'device_screen_ratio'

## Сгенерим доп признаки sqr, sqrt, log на числовые колонки

## Доп признаки по OHE кодированию по колонкам 'device_category', 'car_brand', 'car_model', 'visit_time_night', 'geo_country_is_russia', geo_city_is_big', 'geo_city_distance_from_moscow_category', 'device_browser'

## Создадим train_test_split

## Обучим базовую модель
![image](https://user-images.githubusercontent.com/104129537/234870123-cd92c5de-385e-4f69-b6bc-797081816c72.png)

## Обучим лог регресиию
![image](https://user-images.githubusercontent.com/104129537/234871018-90a4de5e-476a-4dea-911e-56d70f370834.png)

## Обучим метод опорных векторов
![image](https://user-images.githubusercontent.com/104129537/234871230-46435ad8-c94b-419a-ac93-2273f23c0c8c.png)

## Обучим нейронные модели
![image](https://user-images.githubusercontent.com/104129537/234871697-607923c9-fd54-4c95-9acf-08d5aaddc19b.png)

## Обучим Байессовский классификатор
![image](https://user-images.githubusercontent.com/104129537/234871983-24234f04-4e41-4ed1-9b53-cb2de20cc8e0.png)

## Обучим Деревья решений
![image](https://user-images.githubusercontent.com/104129537/234872165-2c38cb44-4e1c-4a94-99dd-1c9f876b1b66.png)

## Обучим модель случайного леса
![image](https://user-images.githubusercontent.com/104129537/234872287-9ae25b67-2858-4f84-a545-39b6846cac88.png)

## Обучим Дерево классификации на основе гистограммного градиентного
![image](https://user-images.githubusercontent.com/104129537/234872483-eddad4a1-48f3-4b9c-b57e-ca1f5ed37d55.png)

## Обучим CatBoost — библиотека для градиентного бустинга, главным преимуществом которой является то, что она одинаково хорошо работает «из коробки» как с числовыми признаками, так и с категориальными
![image](https://user-images.githubusercontent.com/104129537/234872766-05eaf2d7-a200-4eb0-ba9c-f4eca342c279.png)

## Обучим XGBoost 
![image](https://user-images.githubusercontent.com/104129537/234872995-d551506b-5d67-4123-9ed6-918d798e4d61.png)

## Обучим LGBMClassifier  
![image](https://user-images.githubusercontent.com/104129537/234873127-947af564-4e32-4aff-888f-fd9625fe0046.png)


## Оптимизация модели
Лучшей моделью является LGBMClassifier по следующим причинам:

Один из лучших показателей roc_auc.Быстрое обучение.Модель интерпретируема, то есть можно получить показатели важности признаков.
Может предсказывать вероятность класса.Оптимизация модели и конвейера по подготовке данных проводится с помощью байесовской оптимизации. При разных гиперпараметрах модель обучается на тренировочных данных, а оценивается на валидационных.

##  Подбор параметров
![image](https://user-images.githubusercontent.com/104129537/234873452-097d8995-2d61-4bd5-98bd-9c208bf95623.png)

##  Расчет на тренировочных и тестовых данных
![image](https://user-images.githubusercontent.com/104129537/234873609-968b9002-074b-40da-852c-dbb69d3facad.png)


##  ROC-кривая (Receiver operating characteristic)
![image](https://user-images.githubusercontent.com/104129537/234873738-28f0e241-cc68-49ed-aecc-261fce34dd6a.png)

##  Расчет на фин модели
![image](https://user-images.githubusercontent.com/104129537/234874090-c4fa9625-86f6-4e46-ab50-a4e33c4c0e78.png)

##  Влияние фич на модель
![image](https://user-images.githubusercontent.com/104129537/234874231-4d0a76fa-0612-4694-95bd-b8a9dcb1f6d2.png)

