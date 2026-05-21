
import json

try:
    with open('d:/predarb/markets_config.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("JSON is valid.")
    print("Critics Choice Awards events found:")
    critics_count = 0
    for event in data['events']:
        if 'Critics Choice Awards' in event['name']:
            print(f"- {event['name']} (Ticker: {event['kalshi_ticker']})")
            critics_count += 1
    print(f"Total Critics Choice Awards markets found: {critics_count}")
except Exception as e:
    print(f"Error: {e}")
