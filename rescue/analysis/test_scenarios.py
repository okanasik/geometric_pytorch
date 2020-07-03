from sim_wrapper.scenario_runner import ScenarioRunner
import os
import json
import numpy as np
import rescue.dataset.scenario_manager as scn_manager
import random


def get_test_scenarios():
    return ["test{}".format(i) for i in range(500)]


def combine_team_scores():
    all_scores = {}
    with open("team_scores_all.json", "r") as fp:
        team_scores = json.load(fp)
    for team in team_scores:
        all_scores[team] = team_scores[team]
        print(team)

    with open("other_team_scores.json", "r") as fp:
        team_scores = json.load(fp)
    del team_scores["csu"]
    for team in team_scores:
        all_scores[team] = team_scores[team]
        print(team)

    with open("other_team_scores2.json", "r") as fp:
        team_scores = json.load(fp)
    for team in team_scores:
        all_scores[team] = team_scores[team]
        print(team)

    with open("all_team_scores.json", "w") as fp:
        json.dump(all_scores, fp)

    print("----------")
    for team in all_scores:
        print(team)


def combine_results():
    with open("team_scores.json", "r") as fp:
        json_scores = json.load(fp)

    fp = open("ait_data.txt", "r")
    for line in fp:
        line = line.strip()
        if line == "ait":
            team = "ait"
            json_scores[team] = {}
        elif line == "aitsample":
            team = "aitsample"
            json_scores[team] = {}
        elif line.startswith("test"):
            scn = line
            json_scores[team][scn] = {}
            json_scores[team][scn]["final_score"] = 0.0
            json_scores[team][scn]["scores"] = []
        elif line.startswith("["):
            json_scores[team][scn]["scores"] = json.loads(line)
        else:
            json_scores[team][scn]["final_score"] = float(line)

    with open("team_scores_all.json", "w") as fp:
        json.dump(json_scores, fp)


def summarize_results(data_filename, scenarios=None):
    with open(data_filename, "r") as fp:
        json_data = json.load(fp)

    for team in json_data:
        print(team)
        scores = []
        for scn in json_data[team]:
            if scenarios is not None and scn in scenarios:
                print(scn)
                scores.append(json_data[team][scn]["final_score"])
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        print("{} +- {}".format(avg_score, std_score))


def run_scenarios(score_file_name, scenarios, new_seed=False, num_sim_per_scn=1):
    scn_root = "/home/okan/rescuesim/scenarios/test/"
    # teams = ["ait", "mrl", "aitsample", "csu", "rio"]
    teams = ["ait"]
    stats = {}
    for team in teams:
        stats[team] = {}
        print(team)
        for scn in scenarios:
            print(scn)
            stats[team][scn] = {}
            stats[team][scn]["final_score"] = []
            stats[team][scn]["scores"] = []
            for i in range(num_sim_per_scn):
                scn_folder = os.path.join(scn_root, scn)
                if new_seed:
                    original_seed = scn_manager.get_random_seed(scn_folder)
                    new_rnd_seed = random.randrange(1000000)
                    scn_manager.set_random_seed(scn_folder, new_rnd_seed)

                runner = ScenarioRunner(team, "false", scn_folder, viewer="")
                final_score, scores = runner.run()

                if new_seed:
                    scn_manager.set_random_seed(scn_folder, original_seed)

                stats[team][scn]["final_score"].append(final_score)
                stats[team][scn]["scores"].append(scores)
                print(final_score)
                print(scores)

            with open(score_file_name, "w") as outfile:
                json.dump(stats, outfile)


if __name__ == "__main__":
    # run_scenarios()
    # combine_results()
    # summarize_results("aitil_scores.json")
    # summarize_results("all_team_scores.json")
    # combine_team_scores()
    # training_scns = ["test{}".format(i) for i in range(400)]
    # print("ait")
    # summarize_results("ait_scores.json", scenarios=scns)
    # print("aitsample")
    # summarize_results("aitsample_scores.json", scenarios=scns)
    # print("aitil")
    # summarize_results("aitil_topk_notnull_scores.json", scenarios=scns)
    # print("aitil_notrandom")
    # summarize_results("aitil_topk_notnull_notrandom_scores.json", scenarios=scns)
    # print("rio")
    # summarize_results("rio_scores.json", scenarios=scns)
    # print("mrl")
    # summarize_results("mrl_scores.json", scenarios=scns)
    # print("csu")
    # summarize_results("csu_scores.json", scenarios=scns)
    # summarize_results("aitrl_bc_rl_scores.json", scenarios=training_scns)
    # test_scns = ["test{}".format(i) for i in range(400,500)]
    # summarize_results("aitrl_bc_rl_scores.json", scenarios=test_scns)
    run_scenarios("ait_test0_rndseeds_scores.json", ["test0"], new_seed=True, num_sim_per_scn=100)









