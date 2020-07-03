from datetime import datetime
import xml.etree.ElementTree as et


def convert_to_datetime(datetime_str):
    if not datetime_str: return None
    datetime_values = [2020, 1, 1, 0, 0, 0] # [year, month, day, hour, min, sec]
    datetime_values[0] = int(datetime_str[0:4])
    datetime_str = datetime_str[4:]
    idx = 1
    while datetime_str and idx < len(datetime_values):
        datetime_values[idx] = int(datetime_str[0:2])
        datetime_str = datetime_str[2:]
        idx += 1

    return datetime(year=datetime_values[0], month=datetime_values[1], day=datetime_values[2], hour=datetime_values[3],
                    minute=datetime_values[4], second=datetime_values[5])


def read_rescue_gml(file_path):
    tree = et.parse(file_path)
    root = tree.getroot()

    building_ids = []
    for building in root.iter('{urn:roborescue:map:gml}building'):
        bid = building.attrib['{http://www.opengis.net/gml}id']
        building_ids.append(bid)

    road_ids = []
    for road in root.iter('{urn:roborescue:map:gml}road'):
        rid = road.attrib['{http://www.opengis.net/gml}id']
        road_ids.append(rid)

    return building_ids, road_ids


def read_num_agents(scn_file_path):
    tree = et.parse(scn_file_path)
    root = tree.getroot()
    amb_count = 0
    fb_count = 0
    police_count = 0
    civ_count = 0
    for amb in root.iter('{urn:roborescue:map:scenario}ambulanceteam'):
        amb_count += 1
    for fb in root.iter('{urn:roborescue:map:scenario}firebrigade'):
        fb_count += 1
    for police in root.iter('{urn:roborescue:map:scenario}policeforce'):
        police_count += 1
    for civ in root.iter("{urn:roborescue:map:scenario}civilian"):
        civ_count += 1

    return amb_count, fb_count, police_count, civ_count


def kill_process_by_match(matching_string):
    import subprocess
    output = subprocess.run("ps aux", shell=True, capture_output=True).stdout
    process_list = output.decode("utf-8").split("\n")
    for process in process_list:
        if matching_string in process:
            values = [v for v in process.split(" ") if v]
            pid = values[1]
            subprocess.run("kill -9 {}".format(pid), shell=True, capture_output=True)


if __name__ == '__main__':
    # bids, rids = read_rescue_gml('/home/okan/rescuesim/scenarios/robocup2019/test2/map/map.gml')
    # print(bids)    print(process_list)
    # print(rids)

    # print(read_num_agents('/home/okan/rescuesim/scenarios/robocup2019/test2/map/scenario.xml'))
    kill_process_by_match("rcrs-server")