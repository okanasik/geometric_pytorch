from scenario_runner import ScenarioRunner
import timeit
import os


def run_sim():
    scn_root = "/home/okan/rescuesim/scenarios/test/"
    scn_name = "test0"
    runner = ScenarioRunner("ait", "false", os.path.join(scn_root, scn_name), viewer="")
    final_score, scores = runner.run()


if __name__ == "__main__":
    num_run = 10
    avg_time = timeit.timeit(run_sim, number=num_run) / float(num_run)
    print("avg test0 senario time is:{}".format(avg_time))
