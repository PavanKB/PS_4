from gymnasium import Env
import seaborn as sns
import matplotlib.pyplot as plt
from plots import Plots


def update_reward(env: Env, prob_move: float = None, stay_put_cost: float = -1,  move_cost: float = -0.01, goal_reward: float = 100, hole_penalty: float = -100):
    """
    Add a small penalty for each step taken
    Add a massive penalty for falling into a hole
    Add a massive reward for reaching the goal

    # This has been copied from - Michael Ardis Holley
    # https://edstem.org/us/courses/51727/discussion/4714977?answer=10894448

    Args:
        env (Env): The openAi Gym Environment
        stay_put_cost (float, optional): Penalty for staying in place. Defaults to -1.
        move_cost (float, optional): Penalty for each step taken. Defaults to -0.01.
        goal_reward (float, optional): Reward for reaching the goal. Defaults to 100.
        hole_penalty (float, optional): Penalty for falling into a hole. Defaults to -100.


    Returns:
        Env: Updated openAi Gym Environment
    """

    if prob_move is not None:
        assert 0 <= prob_move <= 1, "prob_move must be between 0 and 1"

    for k in env.P.keys():

        # For each node
        v = env.P[k]

        # For each action
        for a in v.keys():
            
            # probability, next_state, reward, done
            
            p, s, r, t = v[a][0]
            if s == k:
                # alter r for staying in-place
                r = stay_put_cost
            elif r == 1.0:
                # alter r for reaching the goal
                r = goal_reward
            elif t and r == 0.0:
                # alter r for falling into a hole
                r = hole_penalty
            else:
                # alter r for moving to another space
                r = move_cost

            if prob_move is not None:
                # update the prob of move
                p = prob_move
            
            env.P[k][a][0] = (p, s, r, t)
            
    return env


def plot_frozen_lake_policy(V, pi: dict, map_size: int, p_frozen: float, gamma: float, n_i: int, iter_type: str):
    """Plots the heatmap and the policy direction for the Frozen Lake Environment

    Args:
        V (np.array): Final value matrix
        pi (_type_): policy matrix
        map_size (int): size of map
        p_frozen (float): prob of frozen tile. (Used for the the title of the plot)
        gamma (float): _description_ (Used for the the title of the plot)
        n_i (int): Number of iteration to get convergence (Used for the the title of the plot)
        iter_type (str): The type of iteration policy/value (Used for the the title of the plot)
    """
    # plot state values
    sns.set_theme(rc={'figure.figsize':(20.7,8.27)})

    fig, ax = plt.subplots(ncols=2)
    plt.subplots_adjust(wspace=0, hspace=0)

    size=(map_size, map_size)

    Plots.values_heat_map(V, "State Values", size, ax[0])

    v_max, directions = Plots.get_policy_map(pi, V, {0: "←", 1: "↓", 2: "→", 3: "↑"}, size)

    Plots.plot_policy(v_max, directions=directions, title='Policy Map', ax=ax[1])

    fig.suptitle(f'Frozen Lake {map_size}x{map_size}, slippery: {True}, p_frozen: {p_frozen}, gamma: {gamma}\n{iter_type} - n_tier:{n_i}', fontsize=16);