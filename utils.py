import requests

API_KEY = "YOUR_API_KEY"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url).json()

    temp = res["main"]["temp"]
    humidity = res["main"]["humidity"]
    rainfall = res.get("rain", {}).get("1h", 0)

    return temp, humidity, rainfall
