import sys
import os
import sqlite3
import panel as pn
import json
from datetime import timedelta, datetime
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from riskMonitoring.pipeline import daily_update
from riskMonitoring.initialize_db import initialize_db
path = os.path.join(os.path.dirname(__file__), "../..", 'instance', 'local.db')

# check if db exists at /instance/local.db, if not, create one
if not os.path.exists(path):
    conn = sqlite3.connect(path)
    conn.close()
    initialize_db()

# create a credential file if not exist
credential_path = os.path.join(os.path.dirname(__file__), "../..", 'instance', 'credential.json')
# create a default username
if not os.path.exists(credential_path):
    with open(credential_path, 'w') as f:
        json.dump(dict(user='password'), f)

# update everyday at market close 15:00
pn.state.schedule_task('task', daily_update, cron='5 15 * * mon,tue,wed,thu,fri')