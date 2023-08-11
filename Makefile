setup:
	pip3 install -r requirements.txt

app:
	python3 app.py

docker-compose:
	docker-compose up --build