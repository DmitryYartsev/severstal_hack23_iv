# Severstal Hack - Сервис для продвинутой аналитики простоев (Команда InfoVision)
**Проект для развертывания сервиса** </br>
**1. Для развертывание сервиса локально, необходимо ввести следующие команды: </br>**
```
git clone https://github.com/DmitryYartsev/severstal_hack23_iv.git
sudo chmod -R 777 severstal_hack23_iv
cd severstal_hack23_iv
sudo bash run_service.sh
```
**2.  Далее необходимо подключить Grafana</br>**
- Зайти на host:3000, ввести admin\admin
- Необходимо нажать на список меню позле вкладки Home (левый верхний угол) и выбрать Connections
- В поле search all ввести postgresql и выбрать БД
- Справа нажать на create a PostgresQL data source
- ввести следующие конфиги: 
- 1. host - 172.27.0.5:5432
- 2. Database - plant_database
- 3. User - user
- 4. Password - pass
- 5. TSL\SSL mode - disable
- Нажать на кнопку Save and test

**3. Далее необходимо загрузить дашборд </br>**
- Необходимо нажать на список меню позле вкладки Home (левый верхний угол) и выбрать Dashboard
- Нажать на кнопку New -> Import -> Upload JSON
- Загрузить туда файл "Дашборды.json"
- Если Grafana выдает ошибку "Datasource not found", нажать на интересующий дашборда, нажать кнопку E, клинктуь на поле с кодом, кликнуть на кнопку run query 