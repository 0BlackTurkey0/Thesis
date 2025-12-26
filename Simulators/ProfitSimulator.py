import math
import random
import matplotlib.pyplot as plt

def profit_percentage(pools: list[int]) -> list[float]:
    if len(pools) < 2: raise ValueError("length of pools must be greater than 1")
    for pool in pools:
        if pool < 0: raise ValueError("element of pools must be positive")

    scaled_pools = [pool ** (1 / math.log2(len(pools)) + 1) for pool in pools]

    sum_pools = sum(pools)
    sum_scaled_pools = sum(scaled_pools)
   
    profit_percentages = [100 * ((scaled_pool / sum_scaled_pools) * (sum_pools / pool) - 1) if pool > 0 else 0 for pool, scaled_pool in zip(pools, scaled_pools)]

    return profit_percentages

# For Testing
def profit_class_plot(pools: list[int], expo: float) -> None:
    if len(pools) < 2: raise ValueError("length of pools must be greater than 1")
    for pool in pools:
        if pool < 0: raise ValueError("element of pools must be positive")
    if expo < 1: raise ValueError("exponent must be greater than or equal to 1")

    num_class = len(pools)
    sum_pools = sum(pools)
    
    scaled_pools = [pool ** (expo) for pool in pools]
    sum_scaled_pools = sum(scaled_pools)
    
    normalized_scaled_pools = [scaled_pool / sum_scaled_pools * sum_pools for scaled_pool in scaled_pools]

    profits = [normalized_scaled_pool - pool for pool, normalized_scaled_pool in zip(pools, normalized_scaled_pools)]

    profit_percentages = [100 * (pool_profit / pool) if pool > 0 else 0 for pool, pool_profit in zip(pools, profits)]

    # profit_percentages_by_func = profit_percentage(pools)
    
    # print(f"pools = {pools}")
    # print(f"scaled_pools = {scaled_pools}")
    # print(f"normalized_scaled_pools = {normalized_scaled_pools}")

    # print(f"profits = {profits}")
    # print(f"profit_percentages = {profit_percentages} (%)")

    # print(f"profit_percentages_by_func = {profit_percentages_by_func} (%)")

    _, ax = plt.subplots()
    names = ["class_" + str(idx) for idx in range(num_class)]
    pools_bar = ax.bar(names, pools, color="darkblue", width=0.8)
    ax.bar_label(pools_bar, [f"{p}\n↓\n{n:.2f}\n({f:+.2f})" for p, n, f in zip(pools, normalized_scaled_pools, profits)], label_type="center", padding=-10, color='w', weight="bold", fontsize=12)
    rewards = [profit if profit > 0 else 0 for profit in profits]
    penalties = [profit if profit < 0 else 0 for profit in profits]
    rewards_bar = ax.bar(names, rewards, bottom=pools, color="lime", width=0.8)
    ax.bar_label(rewards_bar, [f"{p:+.2f}%" if p > 0 else '' for p in profit_percentages], label_type="center", weight="bold", fontsize=12)
    penalties_bar = ax.bar(names, penalties, bottom=pools, color="red", width=0.8)
    ax.bar_label(penalties_bar, [f"{p:+.2f}%" if p < 0 else '' for p in profit_percentages], label_type="center", weight="bold", fontsize=12)
    ax.set_title(f"Profit Percentage Analysis with {num_class} Classes & {sum_pools} Clients (Exponent: {expo})", pad=15, fontsize=22)
    ax.set_xlabel("Classes", labelpad=15, fontsize=20)
    ax.set_ylabel("Tokens (Number of People)", labelpad=15, fontsize=20)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.show()

def profit_time_plot(probs: list[float], iteration: int, init_tokens: float, num_class: int) -> None:
    if len(probs) < 2: raise ValueError("length of probs must be greater than 1")
    for prob in probs:
        if prob < 0 or prob > 1: raise ValueError("element of probs must be between 0 and 1")
    if iteration < 2: raise ValueError("iteration must be greater than 1")
    if init_tokens < 1: raise ValueError("initial tokens must be greater than or equal to 1")
    if num_class < 2: raise ValueError("length of pools must be greater than 1")

    num_type = len(probs)

    tokens_list = [[init_tokens] * 100 for _ in range(num_type)]
    avg_tokens_list = [[init_tokens] * num_type]

    for _ in range(iteration):
        shuffle_idx = [i for i in range(num_class)]
        random.shuffle(shuffle_idx)
        total = 100
        label_distib = [0] * num_class
        for i in range(num_class - 1):
            number = random.randint(math.ceil(total / 2), total)
            label_distib[shuffle_idx[i]] = number
            total -= number
        label_distib[shuffle_idx[-1]] = total
        # print(label_distib)
        
        pools = [0] * num_class
        predicts = [[-1] * 100 for _ in range(num_type)]
        for type_idx in range(num_type):
            for client_idx in range(100):
                if tokens_list[type_idx][client_idx] >= 1:
                    rand = random.random()
                    pred = -1
                    if rand < probs[type_idx]:
                        number = random.randint(1, 100)
                        cnt, idx = 0, -1
                        while number > cnt:
                            idx += 1
                            cnt += label_distib[shuffle_idx[idx]]
                        pred = shuffle_idx[idx]
                    else:
                        pred = random.randint(0, num_class - 1)
                    pools[pred] += 1
                    predicts[type_idx][client_idx] = pred
        # print(pools)
        # print(predicts)

        profit_percentages = profit_percentage(pools)
        # print(profit_percentages)

        for type_idx in range(num_type):
            for client_idx in range(100):
                if tokens_list[type_idx][client_idx] >= 1:
                    tokens_list[type_idx][client_idx] += profit_percentages[predicts[type_idx][client_idx]] / 100
        
        
        avg_tokens = [sum(tokens) / 100 for tokens in tokens_list]
        avg_tokens_list.append(avg_tokens)

    # print(avg_tokens_list)

    _, ax = plt.subplots()
    for type_idx in range(num_type):
        data = [avg_tokens_list[i][type_idx] for i in range(iteration + 1)]
        ax.plot(range(iteration + 1), data, linewidth=3, label=f"Predict with {probs[type_idx] * 100:.0f}% Accuracy")
    ax.legend(loc="upper left", fontsize=15)
    ax.set_title(f"Tokens Analysis with {iteration} Iterations & {num_type} Types of Accuracy (100 Clients per Type) on {num_class} Classes (Fees: {init_tokens})", pad=15, fontsize=22)
    ax.set_xticks(range(0, iteration + 1, math.floor(iteration / 10)))
    ax.set_xlabel("Number of Data Iterations", labelpad=15, fontsize=20)
    ax.set_ylabel("Average Tokens", labelpad=15, fontsize=20)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    pools = [32, 43, 111, 156, 164, 68, 39, 37, 81, 51]
    # pools = [95, 212, 264, 139, 72]
    expo = 1 / math.log2(len(pools)) + 1
    # expo = 3.0
    profit_class_plot(pools, expo)

    probs = [1 / len(pools), 0.3, 0.5, 0.7, 0.8, 0.9]
    # probs = [1 / len(pools), 0.5, 0.75, 0.89, 0.90, 0.91]
    iteration = 10000
    init_tokens = 100
    profit_time_plot(probs, iteration, init_tokens, len(pools))