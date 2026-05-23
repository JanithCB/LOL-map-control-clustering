# Step 5 (Optional): Aggregate snapshot clusters to game-level playstyles
#
# If match-level sequences of minimap frames are available:
#   - Assign each frame to a minimap cluster
#   - For each game, compute distribution of cluster types over early game
#   - Cluster games based on these distributions
#
# Output: game-level playstyle clusters
#   e.g., "aggressive grouping early", "lane-stable early", "bot-side heavy"
