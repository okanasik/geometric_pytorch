import subprocess
import os
import rescue.utils as utils
import uuid


class ScenarioRunner():
    def __init__(self, team, data, scn_path, print_out=False, rescue_path="/home/okan/rescuesim/rcrs-server/boot/", viewer="-v"):
        self.team = team
        self.data = data
        self.scn_path = scn_path
        self.rescue_path = rescue_path
        self.viewer = viewer
        self.print_out = print_out
        self.process_id = None

    def run(self):
        scores = []
        final_score = 0
        config_path = os.path.join(self.scn_path, "config")
        map_path = os.path.join(self.scn_path, "map")
        self.process_id = uuid.uuid4().hex
        script = "cd {} && ./demo.sh -c {} -m {} -t {} -d {} {} -p {}".format(self.rescue_path, config_path, map_path,
                                                                                   self.team, self.data,
                                                                                   self.viewer, self.process_id)
        process = subprocess.Popen(script, shell=True, stdout=subprocess.PIPE)
        for line in iter(process.stdout.readline, ''):
            line = line.decode('utf-8').strip()
            if "time:" in line and "score:" in line:
                if self.print_out: print(line)
                score = float(line.split(" ")[-1].split(":")[-1])
                scores.append(score)
            elif "finalscore" in line:
                final_score = float(line.split(" ")[-1].split(":")[-1])
            elif 'kernelshutdown' in line:
                process.terminate()
                break

        utils.kill_process_by_match(self.process_id)
        return final_score, scores

    def get_num_agents(self):
        return utils.read_num_agents(os.path.join(self.scn_path, "map", "scenario.xml"))


if __name__ == "__main__":
    scn_root = "/home/okan/rescuesim/scenarios/test/"
    scn_name = "test0"
    runner = ScenarioRunner("ait", "false", os.path.join(scn_root, scn_name), viewer="-v")
    final_score, scores = runner.run()
    print(final_score)
    print(scores)
