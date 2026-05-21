import requests
import json

def inspect():
    # Use a generic slug or search for active events
    url = "https://gamma-api.polymarket.com/events"
    params = {"limit": 1, "closed": "false"} 
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            print("No events found")
            return

        event = data[0]
        print(f"Event: {event.get('title')}")
        
        markets = event.get('markets', [])
        if markets:
            m = markets[0]
            print("\nMarket Keys:", m.keys())
            print("\nSample Market Data:")
            print(json.dumps(m, indent=2))
            
    except Exception as e:
        print(e)

inspect()
