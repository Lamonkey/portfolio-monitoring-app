import utils as utils
import json
import os
import datetime as dt

class Log():

    def __init__(self, path):
        current_path = os.path.dirname(os.path.abspath(__file__))
        db_dir = os.path.join(current_path, "../..", 'instance', 'log.json')
        self.path = db_dir
        self.log = self._load_json_file(self.path)

    def _load_json_file(self, path):
        # if not exist return empty dict
        if not os.path.exists(path):
            return {}
        # problem loading file return empty dict
        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
                return data
        except:
            return {}

    def _save_json_file(self, path, data):
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def update_log(self, key):
        self.log[key] = utils.time_in_beijing().strftime('%Y-%m-%d %H:%M:%S')
        self._save_json_file(self.path, self.log)

    def get_time(self, key):
        time_str = self.log.get(key, None)
        if time_str is None:
            return None
        return dt.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    log = Log('instance/log.json')
    log.update_log('stock_update')
    # print(log.log)