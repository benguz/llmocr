# setup

```
pip install virtualenv
python -m venv .venv
.venv\scripts\activate # source .venv/bin/activate for mac/unix
```

next

```
pip install -r requirements.txt
set FLASK_APP=app.py # export FLASK_APP=app.py in Mac
```

create a .env file in /python_selfhost and add:
```
OPENAI_API_KEY='your-api-key-here'
```

you're ready to go!

```
python -m flask run
```
