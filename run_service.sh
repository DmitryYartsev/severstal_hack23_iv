# Создать сеть для контейнеров
docker network create --driver bridge --subnet=172.27.0.0/16 --gateway=172.27.0.1 plant_network
# Создать графану
docker run -dit --name=grafana -p 3000:3000 -u 0 --ip 172.27.0.3 --network plant_network grafana/grafana
# Создать БД PostgresQL
docker run -dit --name postgres_db -e POSTGRES_DB=plant_database -e POSTGRES_USER=user \
	-e POSTGRES_PASSWORD=pass -e PGDATA=/var/lib/postgresql/data/pgdata --ip 172.27.0.5 -p 5432:5432 -u 0 --network plant_network postgres
# Создать PgAdmin
docker run -dit -e PGADMIN_DEFAULT_PASSWORD=pass -e PGADMIN_DEFAULT_EMAIL=email@ex.com --name pgadmin --ip 172.27.0.6 -p 80:80 -u 0 --network plant_network dpage/pgadmin4
# Создать ML python backend
docker run -dit --name python_container -p 8889:8888 -v .:./ --network plant_network -u 0 python_server
# Подождать, пока докеры откроются
sleep 10
# Установки необходимых библиотек
docker exec -it python_container pip3 install -r requirements.txt
# Загрузка данных в БД 
docker exec -it python_container python3 ./db_init.py
# Запуск ML моделей
# docker exec -dit python_container python3 ./run_ml.py
