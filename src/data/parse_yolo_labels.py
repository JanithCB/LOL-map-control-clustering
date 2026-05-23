# Step 1: Parse YOLO label files into champion (x, y) positions
# 
# Reads .txt label files in YOLO format:
#   class_id x_center y_center width height  (all relative 0-1)
#
# Converts to absolute pixel coordinates and normalizes
# to a fixed [0,1] x [0,1] coordinate system.
#
# Output: For each image, a list of (champion_class, x, y) tuples.
