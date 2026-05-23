# Step 2: Define map zones on the minimap
#
# Zones (as polygons/rectangles in normalized [0,1] x [0,1] space):
#   - Top lane
#   - Mid lane
#   - Bot lane
#   - Blue-side jungle (friendly)
#   - Red-side jungle (enemy)
#   - River
#   - Dragon pit
#   - Baron pit
#
# Provides:
#   - Zone definitions loaded from configs/map_zones.yaml
#   - Function to classify a (x, y) point into a zone
#   - Visualization of zone boundaries on a minimap
