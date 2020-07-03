import shutil
import utils
import random
import os


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


def set_random_seed(scn_folder, seed):
    cfg_path = os.path.join(scn_folder, "config")
    cfg_path = os.path.join(cfg_path, "common.cfg")
    lines = []
    with open(cfg_path, "r") as fp:
        for line in fp:
            lines.append(line.strip("\n"))

    with open(cfg_path, "w") as fp:
        for line in lines:
            if line.startswith("random.seed:"):
                fp.write("random.seed: {}\n".format(seed))
            else:
                fp.write(line+"\n")


def get_random_seed(scn_folder):
    cfg_path = os.path.join(scn_folder, "config")
    cfg_path = os.path.join(cfg_path, "common.cfg")
    with open(cfg_path, "r") as fp:
        for line in fp:
            if line.startswith("random.seed:"):
                values = line.split(":")
                return int(values[1])


def create_scenario(base_scenario_path, root_path, scenario_name, num_fires, num_ambs, num_fbs, num_polices, num_civs):
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
    fb_positions = random.choices(road_ids, k=num_fbs)
    amb_positions = random.choices(all_positions, k=num_ambs)
    police_positions = random.choices(all_positions, k=num_polices)

    fire_positions = random.sample(building_ids, k=num_fires)

    create_scenario_file(os.path.join(full_folder_path, 'map/scenario.xml'), refuge_id, civ_positions,
                         amb_positions, fb_positions, police_positions, fire_positions)


if __name__ == "__main__":
    # set_random_seed("/home/okan/rescuesim/scenarios/robocup2019/test", 1024)
    print(get_random_seed("/home/okan/rescuesim/scenarios/robocup2019/test"))