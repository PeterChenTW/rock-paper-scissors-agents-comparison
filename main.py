from kaggle_environments import evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
scores = np.zeros((len(list_names), 1), dtype=int)

for ind_agent_1 in range(len(list_names)):
    print(list_agents[ind_agent_1])
    current_score = evaluate(
        "rps",
        ["agent.py", list_agents[ind_agent_1]],
        configuration={"episodeSteps": 1000}
    )
    print(current_score)
    scores[ind_agent_1, 0] = current_score[0][0]

df_scores = pd.DataFrame(
    scores,
    index=list_names,
    columns=["agent"],
)

print(df_scores)
# plt.figure(figsize=(2, 10))
# sns.heatmap(
#     df_scores, annot=True, cbar=False,
#     cmap="coolwarm", linewidths=1, linecolor="black",
#     fmt="d", vmin=-500, vmax=500,
# )
# plt.xticks(rotation=90, fontsize=15)
# plt.yticks(rotation=360, fontsize=15)