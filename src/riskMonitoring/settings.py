from datetime import timedelta
import os
def get_instance_folder_path():
    cur_file_path = os.path.abspath(__file__)
    return os.path.join(os.path.dirname(cur_file_path), "../..", 'instance')


INSTANCT_FOLDER_PATH = get_instance_folder_path()
stream_frequency = timedelta(seconds=60)
FREQUENCY = timedelta(hours=24)

TABLE_NAME_AND_FREQ = [
    ('benchmark_profile', timedelta(days=1)),
    ('portfolio_profile', timedelta(days=1))
]
COMPONENT_WIDTH = 375

HANDLE_FEE = 1/1000