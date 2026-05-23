# Step 3: Compute spatial features per minimap snapshot
#
# Features per snapshot:
#   - Lane presence: ally/enemy count in top/mid/bot
#   - Roaming: champions in river/jungle vs lanes
#   - Grouping: avg pairwise distance, max group size (3+ within radius)
#   - Side pressure: allies in enemy half, enemies in your half
#   - Objective proximity: allies near dragon/baron
#
# Output feature vector:
#   [ally_top, ally_mid, ally_bot, enemy_top, enemy_mid, enemy_bot,
#    ally_jungle, ally_river, enemy_jungle, enemy_river,
#    ally_max_group, ally_avg_dist, ally_in_enemy_half,
#    enemy_in_your_half, ally_near_dragon, ally_near_baron, ...]
