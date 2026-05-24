
# Cluster Labeling Review Prompt

You are manually reviewing unsupervised clusters of **League of Legends minimap-derived spatial states**.

Your job is to assign a short, human-readable gameplay label to each cluster and write a brief description that explains the cluster in game language. The goal is to make the output understandable to a recruiter, reviewer, or teammate who should be able to grasp the macro pattern immediately.

## What evidence to use

For each cluster, use only these inputs:

- Cluster size / percentage
- Top features
- Representative samples
- Qualitative macro notes

These are the most important sources for interpretation because they tell you:
- how common the cluster is,
- which spatial features define it,
- what concrete examples from the cluster look like,
- and what macro tendencies were already observed.

## Labeling principles

### Use gameplay language, not ML language

Prefer labels like:

- Standard laning
- Bot-side objective setup
- Top-side objective setup
- 5-man mid group
- Deep invade
- River contest setup
- Objective collapse

Avoid labels like:

- Cluster 2 behavior
- Spatial centroid type A
- PCA-separated region
- Mixed feature state

The final label should sound like a **macro playstyle or map state**, not a technical artifact.

### Keep labels short

A good label should usually be 2 to 5 words.

Good:
- Standard laning
- Deep invade
- 5-man mid group
- Bot-side dragon setup

Less good:
- Distributed early-game low-pressure lane equilibrium state

### Use the description to explain the cluster

The label should be short. The description should carry the detail.

Example:
- **Label:** Bot-side objective setup
- **Description:** Champions are concentrated in bot lane, bot river, and dragon-side jungle, suggesting setup for dragon control or pressure before a contest.

## How to interpret the evidence

### 1. Cluster size / percentage
Ask:
- Is this a very common cluster?
- Is it likely a default/neutral map state?
- Is it a rarer special-case macro pattern?

Typical interpretation:
- A **very large** cluster often suggests a default state such as standard laning or neutral map distribution.
- A **smaller** cluster may capture a more specialized event such as an invade, grouped push, or objective setup.

### 2. Top features
Ask:
- Which zones dominate: bot, top, mid, river, jungle, pit, objective areas?
- Is the cluster lane-heavy, river-heavy, jungle-heavy, or grouped around an objective?
- Do the defining features indicate spread or concentration?

Examples:
- Strong **bot + river + dragon** features may suggest bot-side objective setup.
- Strong **top + herald/baron + top river** features may suggest top-side objective setup.
- Strong **mid + grouped** features may suggest 5-man mid group.
- Strong **enemy jungle / deep jungle** features may suggest invade behavior.

### 3. Representative samples
Ask:
- What do the sample frames actually look like?
- Are players spread in lanes or grouped?
- Is the cluster visually consistent with the top-feature story?

Representative examples matter because they show the closest concrete examples of the cluster pattern, not just the averaged centroid profile.

### 4. Qualitative macro notes
Ask:
- Do the notes reinforce the feature interpretation?
- Do they mention grouped mid, dragon setup, herald pressure, invade patterns, or river control?
- Do the notes help resolve ambiguity when the top features alone are unclear?

Use notes to strengthen or refine the final gameplay wording.

## Important warning about PCA

Do **not** over-rely on PCA.

PCA is useful for visualization and for checking whether clusters appear separated in 2D, but it is not enough by itself to determine the semantic meaning of a cluster. Final naming should come primarily from:

- top features,
- representative samples,
- cluster size,
- and qualitative macro notes.

Use PCA only as a supporting visual sanity check, not as the main evidence for naming.

## Suggested label patterns

These are examples, not strict rules:

- **Standard laning** — large neutral/default lane-distributed state
- **Bot-side objective setup** — bot lane, bot river, dragon-side concentration
- **Top-side objective setup** — top lane, herald/baron-side concentration
- **5-man mid group** — strong grouped mid presence, low side-lane spread
- **Deep invade** — enemy jungle occupation, aggressive cross-map entry
- **River contest setup** — grouped or contested river presence
- **Objective collapse** — multiple players converging tightly around a key neutral objective

If evidence is weak or ambiguous, choose the best gameplay label you can and lower confidence.

## Confidence guidance

Use:
- **High** — evidence is clear and consistent across size, features, samples, and notes
- **Medium** — the likely interpretation is reasonable, but one source is weak or ambiguous
- **Low** — the cluster is hard to interpret cleanly and needs more manual checking

## Review workflow

For each row in `cluster_interpretation_sheet.csv`:

1. Read the cluster size and percentage.
2. Read the top features.
3. Read the representative samples.
4. Read the macro notes.
5. Assign a short gameplay label.
6. Write a 1–2 sentence description.
7. Assign a confidence rating.
8. Record the reasoning using the template below.

## Exact output template

Use this exact structure for every cluster:

```text
Cluster <id>
Label: <short gameplay label>
Description: <1-2 sentence gameplay interpretation>
Confidence: <high|medium|low>
Reasoning:
- Size: <what the size suggests>
- Top features: <what the strongest features suggest>
- Representative samples: <what the sample frames appear to show>
- Macro notes: <how the qualitative notes support or refine the interpretation>
```

## Example

```text
Cluster 3
Label: Bot-side objective setup
Description: This cluster shows strong bot-side concentration around lane, river, and dragon-side jungle, suggesting setup for dragon pressure or an upcoming contest. It appears more coordinated and objective-focused than a neutral laning state.
Confidence: high
Reasoning:
- Size: Mid-sized cluster, so it looks more specialized than a default state.
- Top features: Bot lane, bot river, and dragon-area features are all strong.
- Representative samples: Frames show multiple players leaning toward bot-side rather than staying evenly distributed.
- Macro notes: Notes mention dragon pressure and grouped bot-side movement, which matches the feature pattern.
```

## Final instruction

When two labels seem possible, choose the one that best describes the **actual gameplay pattern a human would recognize on the minimap**.