from rescue.scenario_runner import ScenarioRunner
import os
import json
import numpy as np


def get_test_scenarios():
    return ["test{}".format(i) for i in range(300, 400)]


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


def summarize_results():
    with open("all_team_scores.json", "r") as fp:
        json_data = json.load(fp)

    for team in json_data:
        print(team)
        scores = []
        for scn in json_data[team]:
            scores.append(json_data[team][scn]["final_score"])
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        print("{} +- {}".format(avg_score, std_score))


if __name__ == "__main__":
    # scenarios = get_test_scenarios()
    # scn_root = "/home/okan/rescuesim/scenarios/robocup2019/"
    # teams = ["csu", "rio"]
    # # teams = ["aitil"]
    # stats = {}
    # for team in teams:
    #     stats[team] = {}
    #     print(team)
    #     for scn in scenarios:
    #         print(scn)
    #         runner = ScenarioRunner(team, "false", os.path.join(scn_root, scn))
    #         final_score, scores = runner.run()
    #         stats[team][scn] = {}
    #         stats[team][scn]["final_score"] = final_score
    #         stats[team][scn]["scores"] = scores
    #         print(final_score)
    #         print(scores)
    #
    #         with open("other_team_scores2.json", "w") as outfile:
    #             json.dump(stats, outfile)
    # combine_results()
    summarize_results()
    # combine_team_scores()







