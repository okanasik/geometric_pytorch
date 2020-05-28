import subprocess
import os


class ScenarioRunner():
    def __init__(self, team, data, scn_path, print_out=False, rescue_path="/home/okan/rescuesim/rcrs-server/boot/", viewer=""):
        self.team = team
        self.data = data
        self.scn_path = scn_path
        self.rescue_path = rescue_path
        self.viewer = viewer
        self.print_out = print_out

    def run(self):
        scores = []
        final_score = 0
        config_path = os.path.join(self.scn_path, "config")
        map_path = os.path.join(self.scn_path, "map")
        script = "cd {} && ./demo.sh -c {} -m {} -t {} -d {} {}".format(self.rescue_path, config_path, map_path,
                                                                                   self.team, self.data,
                                                                                   self.viewer)
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

        return final_score, scores


if __name__ == "__main__":
    scn_root = "/home/okan/rescuesim/scenarios/robocup2019/"
    scn_name = "test300"
    runner = ScenarioRunner("aitsample", "false", os.path.join(scn_root, scn_name))
    final_score, scores = runner.run()
    print(final_score)
    print(scores)
