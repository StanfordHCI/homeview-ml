import json

my_path = './vh.cameras/raw-shorter-JSON-oct-22'

for i in range(1, 128):
    sensors = [sensor for sensor in json.load(open(my_path + '/{}.json'.format(i))) if (sensor['class_name'] == 'light' or sensor['class_name'] == 'door')]
    with open(my_path + '/{}.json'.format(i), 'w') as file_obj:  # open the file in write mode
        json.dump(sensors, file_obj)
