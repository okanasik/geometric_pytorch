import os
from sim_wrapper.scenario_runner import ScenarioRunner


def create_dataset():
    for i in range(500):
        scn_root = "/home/okan/rescuesim/scenarios/test/"
        scn_name = "test{}".format(i)
        runner = ScenarioRunner("ait", "true", os.path.join(scn_root, scn_name), viewer="")
        final_score, scores = runner.run()
        print(scn_name)
        print(final_score)
        print(scores)


if __name__ == '__main__':
    # for i in range(500):
    #     create_scenario('/home/okan/rescuesim/scenarios/robocup2019/test', '/home/okan/rescuesim/scenarios/test',
    #                 'test'+str(i), 10)

    # run_scenario('/home/okan/rescuesim/scenarios/robocup2019', 'test299', '/home/okan/rescuesim/rcrs-server/boot', 'ait')

    # for i in range(300, 400):
    #     print("Running scenario: {}".format(i))
    #     run_scenario('/home/okan/rescuesim/scenarios/robocup2019', 'test'+str(i),
    #                  '/home/okan/rescuesim/rcrs-server/boot', 'ait')
    create_dataset()