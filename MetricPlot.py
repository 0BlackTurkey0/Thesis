import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

'''
is_iid: True or False
anomaly_cnt: 0, 1, 10, 25
threat: "DirtyLabel", "CleanLabel", "Backdoor", "ModelManipulation"
metric: "Loss", "Accuracy", "Precision", "F1Score"
'''
def plot_approaches_comparison(is_iid, anomaly_cnt, threat, metric, legend_loc="lower right"):
    fl2edl_round_rate = 50000 / 1500
    aggrs = ["fed_avg", "l2_norm", "trimmed_mean", "multi_krum"]
    iid_loc = "IID" if is_iid else "NonIID"
    threat_loc = f"{anomaly_cnt}_Anomaly_{threat}" if anomaly_cnt > 0 else f"{anomaly_cnt}_Anomaly"
    threats_names = {"DirtyLabel": "Dirty Label", "CleanLabel": "Clean Label", "Backdoor": "Backdoor", "ModelManipulation": "Model Manipulation"}
    metric_symbol = {
        "Loss": "loss",
        "Accuracy": "acc",
        "Precision": "macro_precision",
        "F1Score": "macro_f1"
    }
    col = f"test_{metric_symbol[metric]}"
    backdoor = "_backdoor" if threat == "Backdoor" and anomaly_cnt > 0 else ""
    df_edl = pd.read_csv(f"./Results/{iid_loc}/{threat_loc}/EDLSave_Results.csv")
    df_fl = {aggr: pd.read_csv(f"./Results/{iid_loc}/{threat_loc}/FLSave_{aggr}_Results.csv") for aggr in aggrs}
    edl_offline = [df_edl[f"{col}_offline{backdoor}"][int(r * fl2edl_round_rate)] for r in range(0, int(-(-len(df_edl[f"{col}_offline{backdoor}"]) / fl2edl_round_rate // 1)))]
    edl_online = [df_edl[f"{col}_online{backdoor}"][int(r * fl2edl_round_rate)] for r in range(0, int(-(-len(df_edl[f"{col}_online{backdoor}"]) / fl2edl_round_rate // 1)))]
    _, ax = plt.subplots()
    ax.plot(range(0, int(-(-len(df_edl[f"{col}_offline{backdoor}"]) / fl2edl_round_rate // 1))), edl_offline, marker=Line2D.filled_markers[1], markersize=10, linewidth=3, label=f"EDL_offline")
    ax.plot(range(0, int(-(-len(df_edl[f"{col}_online{backdoor}"]) / fl2edl_round_rate // 1))), edl_online, marker=Line2D.filled_markers[2], markersize=10, linewidth=3, label=f"EDL_online")
    for aggr, mark in zip(aggrs, Line2D.filled_markers[-4:]):
        fl = [df_fl[aggr][f"{col}{backdoor}"][r] for r in range(0, len(df_fl[aggr][f"{col}{backdoor}"]))]
        ax.plot(range(1, len(df_fl[aggr][f"{col}{backdoor}"]) + 1), fl, marker=mark, markersize=10, linewidth=3, label=f"FL_{aggr}")
    ax.set_title(f"{metric} Evaluation against {anomaly_cnt}% {threats_names[threat] if anomaly_cnt > 0 else "Anomaly"} in {iid_loc}", pad=15, fontsize=18)
    ax.set_xticks(range(0, max(int(-(-len(df_edl.index) / fl2edl_round_rate // 1)), len(df_fl[aggrs[0]].index)) + 1))
    ax.set_xlabel("Number of Rounds (50000 Samples per Round)", labelpad=15, fontsize=15)
    ax.set_ylabel("Test Accuracy", labelpad=15, fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc=legend_loc, fontsize=15)
    ax.grid(True)
    plt.show()

'''
is_iid: True or False
anomaly_cnt: 1, 10, 25
appoach: "EDL", "FL"
varient: "offline", "online", "fed_avg", "l2_norm", "trimmed_mean", "multi_krum"
metric: "Loss", "Accuracy", "Precision", "F1Score"
'''
def plot_threats_comparison(is_iid, anomaly_cnt, appoach, varient, metric, legend_loc="lower right"):
    iid_loc = "IID" if is_iid else "NonIID"
    metric_symbol = {
        "Loss": "loss",
        "Accuracy": "acc",
        "Precision": "macro_precision",
        "F1Score": "macro_f1"
    }
    threats = ["Anomaly_DirtyLabel", "Anomaly_CleanLabel", "Anomaly_Backdoor", "Anomaly_ModelManipulation"]
    threats_names = {"Anomaly_DirtyLabel": "Dirty Label", "Anomaly_CleanLabel": "Clean Label", "Anomaly_Backdoor": "Backdoor", "Anomaly_ModelManipulation": "Model Manipulation"}
    col = f"test_{metric_symbol[metric]}"
    if appoach == "EDL":
        df = {threat: pd.read_csv(f"./Results/{iid_loc}/{anomaly_cnt}_{threat}/EDLSave_Results.csv") for threat in threats}
        col += f"_{varient}"
        round_rate = 50000 / 1500
        offset = 0
    else:
        df = {threat: pd.read_csv(f"./Results/{iid_loc}/{anomaly_cnt}_{threat}/FLSave_{varient}_Results.csv") for threat in threats}
        round_rate = 1
        offset = 1
    _, ax = plt.subplots()
    for threat, mark in zip(threats, Line2D.filled_markers[-4:]):
        backdoor = "_backdoor" if "Backdoor" in threat else ""
        data = [df[threat][f"{col}{backdoor}"][int(r * round_rate)] for r in range(0, int(-(-len(df[threat][f"{col}{backdoor}"]) / round_rate // 1)))]
        ax.plot(range(offset, int(-(-len(df[threat][f"{col}{backdoor}"]) / round_rate // 1)) + offset), data, marker=mark, markersize=10, linewidth=3, label=f"{threats_names[threat]}")
    ax.set_title(f"{metric} Evaluation of {appoach}_{varient} against {anomaly_cnt}% Anomaly in {iid_loc}", pad=15, fontsize=18)
    ax.set_xticks(range(offset, int(-(-len(df[threats[0]].index) / round_rate // 1)) + offset))
    ax.set_xlabel("Number of Rounds (50000 Samples per Round)", labelpad=15, fontsize=15)
    ax.set_ylabel("Test Accuracy", labelpad=15, fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc=legend_loc, fontsize=15)
    ax.grid(True)
    plt.show()

'''
is_iid: True or False
threat: "DirtyLabel", "CleanLabel", "Backdoor", "ModelManipulation"
appoach: "EDL", "FL"
varient: "offline", "online", "fed_avg", "l2_norm", "trimmed_mean", "multi_krum"
metric: "Loss", "Accuracy", "Precision", "F1Score"
'''
def plot_anomaly_cnt_comparison(is_iid, threat, appoach, varient, metric, legend_loc="lower right"):
    iid_loc = "IID" if is_iid else "NonIID"
    threats_names = {"DirtyLabel": "Dirty Label", "CleanLabel": "Clean Label", "Backdoor": "Backdoor", "ModelManipulation": "Model Manipulation"}
    metric_symbol = {
        "Loss": "loss",
        "Accuracy": "acc",
        "Precision": "macro_precision",
        "F1Score": "macro_f1"
    }
    anomaly_cnts = [0, 1, 10, 25]
    col = f"test_{metric_symbol[metric]}"
    if appoach == "EDL":
        df = {anomaly_cnt: pd.read_csv(f"./Results/{iid_loc}/{anomaly_cnt}_Anomaly{f"_{threat}" if anomaly_cnt > 0 else ""}/EDLSave_Results.csv") for anomaly_cnt in anomaly_cnts}
        col += f"_{varient}"
        round_rate = 50000 / 1500
        offset = 0
    else:
        df = {anomaly_cnt: pd.read_csv(f"./Results/{iid_loc}/{anomaly_cnt}_Anomaly{f"_{threat}" if anomaly_cnt > 0 else ""}/FLSave_{varient}_Results.csv") for anomaly_cnt in anomaly_cnts}
        round_rate = 1
        offset = 1
    _, ax = plt.subplots()
    for anomaly_cnt, mark in zip(anomaly_cnts, Line2D.filled_markers[-4:]):
        backdoor = "_backdoor" if threat == "Backdoor" and anomaly_cnt > 0 else ""
        data = [df[anomaly_cnt][f"{col}{backdoor}"][int(r * round_rate)] for r in range(0, int(-(-len(df[anomaly_cnt][f"{col}{backdoor}"]) / round_rate // 1)))]
        ax.plot(range(offset, int(-(-len(df[anomaly_cnt][f"{col}{backdoor}"]) / round_rate // 1)) + offset), data, marker=mark, markersize=10, linewidth=3, label=f"{anomaly_cnt}% {threats_names[threat]}")
    ax.set_title(f"{metric} Evaluation of {appoach}_{varient} against {threats_names[threat]} in {iid_loc}", pad=15, fontsize=18)
    ax.set_xticks(range(offset, int(-(-len(df[anomaly_cnts[0]].index) / round_rate // 1)) + offset))   
    ax.set_xlabel("Number of Rounds (50000 Samples per Round)", labelpad=15, fontsize=15)
    ax.set_ylabel("Test Accuracy", labelpad=15, fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc=legend_loc, fontsize=15)
    ax.grid(True)
    plt.show()

'''
is_iid: True or False
anomaly_cnt: 1, 10, 25
threat: "DirtyLabel", "CleanLabel", "Backdoor", "ModelManipulation"
'''
def plot_tokens(is_iid, anomaly_cnt, threat, legend_loc="lower right"):
    iid_loc = "IID" if is_iid else "NonIID"
    threat_loc = f"{anomaly_cnt}_Anomaly_{threat}" if anomaly_cnt > 0 else f"{anomaly_cnt}_Anomaly"
    threats_names = {"DirtyLabel": "Dirty Label", "CleanLabel": "Clean Label", "Backdoor": "Backdoor", "ModelManipulation": "Model Manipulation"}
    df_edl = pd.read_csv(f"./Results/{iid_loc}/{threat_loc}/EDLSave_Results.csv")
    edl_ordinary_tokens = [df_edl["avg_ordinary_tokens"][r] for r in range(0, len(df_edl["avg_ordinary_tokens"]))]
    edl_anomaly_tokens = [df_edl["avg_anomaly_tokens"][r] for r in range(0, len(df_edl["avg_anomaly_tokens"]))]
    edl_ordinary_clients = [df_edl["active_ordinary_count"][r] for r in range(0, len(df_edl["active_ordinary_count"]))]
    edl_anomaly_clients = [df_edl["active_anomaly_count"][r] for r in range(0, len(df_edl["active_anomaly_count"]))]
    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    axs[0].plot(range(0, int(len(df_edl["avg_ordinary_tokens"]))), edl_ordinary_tokens, linewidth=3, label=f"Ordinary Clients")
    axs[0].plot(range(0, int(len(df_edl["avg_anomaly_tokens"]))), edl_anomaly_tokens, linewidth=3, label=f"Anomaly Clients")
    axs[1].plot(range(0, int(len(df_edl["active_ordinary_count"]))), edl_ordinary_clients, linewidth=3, label=f"Ordinary Clients")
    axs[1].plot(range(0, int(len(df_edl["active_anomaly_count"]))), edl_anomaly_clients, linewidth=3, label=f"Anomaly Clients")
    axs[0].set_title(f"EDL Tokens Evaluation of Active Clients against {anomaly_cnt}% {threats_names[threat] if anomaly_cnt > 0 else "Anomaly"} in {iid_loc}", pad=15, fontsize=18)
    axs[1].set_xlabel("Number of Batches (1000 Samples per Batch)", labelpad=15, fontsize=15)
    axs[0].set_ylabel("Average Tokens", labelpad=15, fontsize=15)
    axs[1].set_ylabel("Number of Active Clients", labelpad=15, fontsize=15)
    axs[0].tick_params(axis='x', labelsize=12)
    axs[0].tick_params(axis='y', labelsize=12)
    axs[1].legend(loc=legend_loc, fontsize=15)
    axs[0].grid(True)
    axs[1].grid(True)
    plt.show()

if __name__ == "__main__":
    plot_approaches_comparison(False, 25, "Backdoor", "Accuracy")
    plot_threats_comparison(True, 25, "FL", "trimmed_mean", "Accuracy", "center right")
    plot_anomaly_cnt_comparison(False, "ModelManipulation", "EDL", "offline", "Accuracy")
    plot_tokens(False, 25, "CleanLabel", "center right")

    # _, ax = plt.subplots()
    # epochs = 200
    # data = [0.3264, 0.6967, 0.8368, 0.8771, 0.8912, 0.9071, 0.9262, 0.9332, 0.9340, 0.9412, 0.9423, 0.9478, 0.9524, 0.9499, 0.9532, 0.9527, 0.9562, 0.9583, 0.9598, 0.9612, 0.9588]
    # ax.plot(range(0, epochs + 1, 10), data, color="lime", marker='o', linewidth=3, label=f"Validation Accuracy")
    # ax.set_title(f"CoAtNet-0 Validation Accuracy on Cifar10", pad=15, fontsize=18)
    # ax.set_xticks(range(0, epochs + 1, 10))
    # ax.set_xlabel("Number of Epochs", labelpad=15, fontsize=15)
    # ax.set_ylabel("Validation Accuracy", labelpad=15, fontsize=15)
    # ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)
    # for x, y in zip(range(0, epochs + 1, 10), data):
    #     ax.text(x, y + 0.01, str(y), ha="center", fontsize=10, weight="bold")
    # ax.grid(True)
    # plt.show()

    # _, ax = plt.subplots()
    # rounds = 100
    # data_mean = [0.7214, 0.7929, 0.8026, 0.8082, 0.8133, 0.8161, 0.8197, 0.8217, 0.8223, 0.8242, 0.8255]
    # data_max = [0.7408, 0.8004, 0.8075, 0.8142, 0.8183, 0.8212, 0.8262, 0.8267, 0.8284, 0.8299, 0.8309]
    # data_min = [0.7013, 0.7853, 0.7962, 0.8017, 0.8073, 0.8107, 0.8145, 0.8163, 0.8180, 0.8184, 0.8204]
    # ax.axhline(0.8296, color="green", linestyle="--", linewidth=3, label=f"Centralized Learning")
    # ax.plot(range(0, rounds + 1, 10), data_mean, color="red", linewidth=3, label=f"Ensemble Distributed Learning (Ours)")
    # ax.fill_between(range(0, rounds + 1, 10), data_max, data_min, color="red", alpha=0.2)
    # ax.set_title(f"Transfer Learning on Cifar10", pad=15, fontsize=18)
    # ax.set_xticks(range(0, rounds + 1, 10))
    # ax.set_xlim(0, 100)
    # ax.set_xlabel("Number of Rounds (500 Data Iterations per Round)", labelpad=15, fontsize=15)
    # ax.set_ylabel("Test Accuracy", labelpad=15, fontsize=15)
    # ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)
    # ax.legend(loc="lower right", fontsize=15)
    # ax.grid(True)
    # plt.show()