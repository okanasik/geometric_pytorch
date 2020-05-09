import os
import shutil
import utils
import random
import subprocess


# global parameters
num_civs = 20
num_fbs = 5
num_ambs = 1
num_polices = 1


def create_scenario_file(file_name, refuge_id, civ_positions, amb_positions, fb_positions, police_positions, fire_positions):
    fp = open(file_name, 'w')
    fp.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
    fp.write('<scenario:scenario xmlns:scenario="urn:roborescue:map:scenario">\n')
    fp.write('<scenario:refuge scenario:location="{}"/>\n'.format(refuge_id))
    for fire_pos in fire_positions:
        fp.write('<scenario:fire scenario:location="{}"/>\n'.format(fire_pos))

    for civ_pos in civ_positions:
        fp.write('<scenario:civilian scenario:location="{}"/>\n'.format(civ_pos))

    for amb_pos in amb_positions:
        fp.write('<scenario:ambulanceteam scenario:location="{}"/>\n'.format(amb_pos))

    for fb_pos in fb_positions:
        fp.write('<scenario:firebrigade scenario:location="{}"/>\n'.format(fb_pos))

    for police_pos in police_positions:
        fp.write('<scenario:policeforce scenario:location="{}"/>\n'.format(police_pos))

    fp.write('</scenario:scenario>\n')

    fp.close()

def create_scenario(base_scenario_path, root_path, scenario_name, num_fires):
    full_folder_path = os.path.join(root_path, scenario_name)
    if os.path.exists(full_folder_path):
        shutil.rmtree(full_folder_path)

    shutil.copytree(base_scenario_path, full_folder_path)

    building_ids, road_ids = utils.read_rescue_gml(os.path.join(full_folder_path, 'map/map.gml'))

    # choose refuge
    refuge_id = random.choice(building_ids)
    building_ids.remove(refuge_id)

    # choose civilian positions
    civ_positions = random.choices(building_ids, k=num_civs)

    # choose agent positions
    all_positions = building_ids + road_ids
    fb_positions = random.choices(all_positions, k=num_fbs)
    amb_positions = random.choices(all_positions, k=num_ambs)
    police_positions = random.choices(all_positions, k=num_polices)

    fire_positions = random.sample(building_ids, k=num_fires)

    create_scenario_file(os.path.join(full_folder_path, 'map/scenario.xml'), refuge_id, civ_positions,
                         amb_positions, fb_positions, police_positions, fire_positions)


def run_scenario(scenario_root_path, scenario, rescue_path, team_name):
    script = "cd " + rescue_path + " && ./demo.sh -c " + os.path.join(scenario_root_path, scenario) + "/config" +\
    " -m " + os.path.join(scenario_root_path, scenario) + "/map -t " + team_name
    num_agents = utils.read_num_agents(os.path.join(scenario_root_path, scenario) + '/map/scenario.xml')
    print(script)
    process = subprocess.Popen(script, shell=True, stdout=subprocess.PIPE)
    num_agent_completed = 0
    for line in iter(process.stdout.readline, ''):
        line = line.decode('utf-8').strip()
        if ': file is saved!' in line:
            num_agent_completed += 1
        if num_agent_completed == num_agents:
            process.terminate()
            break

    # with subprocess.Popen() as process:
    #     print(process.stdout.read())
    # subprocess.run(script, shell=True, stdout=None)


if __name__ == '__main__':
    # for i in range(300, 400):
    #     create_scenario('/home/okan/rescuesim/scenarios/robocup2019/test2', '/home/okan/rescuesim/scenarios/robocup2019',
    #                 'test'+str(i), 7)

    # run_scenario('/home/okan/rescuesim/scenarios/robocup2019', 'test299', '/home/okan/rescuesim/rcrs-server/boot', 'ait')

    for i in range(300, 400):
        print("Running scenario: {}".format(i))
        run_scenario('/home/okan/rescuesim/scenarios/robocup2019', 'test'+str(i),
                     '/home/okan/rescuesim/rcrs-server/boot', 'ait')