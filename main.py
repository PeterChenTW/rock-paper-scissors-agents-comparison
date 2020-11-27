from kaggle_environments import evaluate
import numpy as np
import pandas as pd


list_names = [
    "hit_the_last_own_action",
    "copy_opponent",
    "reactionary",
    "counter_reactionary",
    "statistical",
    "nash_equilibrium",
    "markov_agent",
    "memory_patterns",
    # "multi_armed_bandit",
    "opponent_transition_matrix",
    "decision_tree_classifier",
    "statistical_prediction",
]
list_agents = [agent_name + ".py" for agent_name in list_names]
simulation_times = 10
scores = np.zeros((len(list_names), simulation_times), dtype=int)

for i in range(simulation_times):
    for ind_agent_1 in range(len(list_names)):
        current_score = evaluate(
            "rps",
            ["multi_armed_bandit.py", list_agents[ind_agent_1]],
            configuration={"episodeSteps": 1000}
        )
        print(i, list_names[ind_agent_1], current_score[0][0])
        if current_score[0][0] >= 20:
            add_score = 1
        elif current_score[0][0] <= -20:
            add_score = -1
        else:
            add_score = 0
        scores[ind_agent_1, i] = add_score

df_scores = pd.DataFrame(scores)
df_scores.index = list_names
print(df_scores.mean(axis=1))
# print(df_scores.std(axis=1))
print(df_scores.median(axis=1))
# plt.figure(figsize=(2, 10))
# sns.heatmap(
#     df_scores, annot=True, cbar=False,
#     cmap="coolwarm", linewidths=1, linecolor="black",
#     fmt="d", vmin=-500, vmax=500,
# )
# plt.xticks(rotation=90, fontsize=15)
# plt.yticks(rotation=360, fontsize=15)