# Load and clean ranked games data (games.csv, training_data.csv)
#
# Handles:
#   - Loading EUW ranked games (~50k matches)
#   - Loading NA training data (optional, 10k matches, 700+ features)
#   - Basic cleaning: missing values, type casting
#   - Champion ID mapping using champion_info.json
