from ..config_loader import config # Assuming config_loader is in the parent directory

def load_filters_from_config():
    """
    Loads HSV filter definitions from the global configuration.
    Transforms HSV range lists from config (lists) to tuples as expected by HSVClassifier.
    """
    configured_filters = config.get('hsv_filters', {}).get('team_jersey_filters', [])

    processed_filters = []
    for team_filter_config in configured_filters:
        processed_colors = []
        for color_entry in team_filter_config.get('colors', []):
            processed_colors.append({
                "name": color_entry['name'], # Descriptive name from config
                "lower_hsv": tuple(color_entry['lower_hsv']),
                "upper_hsv": tuple(color_entry['upper_hsv']),
                "weight": color_entry.get("weight", 1.0) # Add this line
            })

        processed_filters.append({
            "name": team_filter_config['name'], # This is the Team Name
            "colors": processed_colors,
        })
    return processed_filters

# Load filters at module import time
filters = load_filters_from_config()

# You can add a function here to get a specific filter by team name if needed later
# def get_filter_for_team(team_name: str):
#     for f in filters:
#         if f['name'] == team_name:
#             return f
#     return None
