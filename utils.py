import requests

API_KEY = "7e2b0e1be93cfa497d0518e26af54841"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url).json()

    temp = res["main"]["temp"]
    humidity = res["main"]["humidity"]
    rainfall = res.get("rain", {}).get("1h", 0)

    return temp, humidity, rainfall
