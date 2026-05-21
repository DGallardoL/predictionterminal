import json
import os

CONFIG_PATH = r"d:\predarb\markets_config.json"
NEW_MARKETS_PATH = r"d:\predarb\temp_new_markets.json"

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: {CONFIG_PATH} not found.")
        return
    if not os.path.exists(NEW_MARKETS_PATH):
        print(f"Error: {NEW_MARKETS_PATH} not found.")
        return

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        with open(NEW_MARKETS_PATH, 'r', encoding='utf-8') as f:
            new_markets = json.load(f)
            
        print(f"Loaded {len(new_markets)} new markets.")
        
        if 'events' not in config:
            config['events'] = []
            
        # Append new markets
        config['events'].extend(new_markets)
        
        # Write back
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully added {len(new_markets)} events to {CONFIG_PATH}")
        
    except Exception as e:
        print(f"Error merging files: {e}")

if __name__ == "__main__":
    main()
