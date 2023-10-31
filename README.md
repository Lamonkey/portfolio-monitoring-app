# Portfolio monitoring app

A web application to monitor portfolio's performance
<img width="2772" alt="image" src="https://github.com/Lamonkey/portfolio-monitoring-app/assets/10794555/cd94cdfb-f329-41b0-9a13-7716207e42ee">


# Try it yourself using docker

> The web app currently only supports Chinese equities, and it is using [jqsdk](https://www.joinquant.com/user/login/index?redirect=/default/index/sdk) to acquire data thus you need to have a free account with them.

## Apple Silicon

```
docker pull lamonkey/risk-monitor-app:arm64
docker volume create rma_volumne
docker run -p 7860:7860 -e JQDATA_USER=USERNAME -e JQDATA_PASSWORD=PASSWORD -e SECRET_COOKIE=my_super_safe_cookie_secret -v rma_volume:/code/instance lamonkey/risk-monitor-app:arm64
```

Go to localhost:7860
The default username and password is user and password

## Intel Machine

```
docker pull lamonkey/risk-monitor-app:amd64
docker volume create rma_volumne
docker run -p 7860:7860 -e JQDATA_USER=USERNAME -e JQDATA_PASSWORD=PASSWORD -e SECRET_COOKIE=my_super_safe_cookie_secret -v rma_volume:/code/instance lamonkey/risk-monitor-app:amd64
```

# Currently Supported Analysis

- [X] Risk
- [X] Compound return
- [X] PnL
- [X] Cash position
- [X] [BHB Return Attribute](https://www.cfainstitute.org/-/media/documents/support/programs/cipm/2019-cipm-l1v1r5.ashx#:~:text=3.1%20The%20Brinson%E2%80%93Hood%E2%80%93Beebower%20(BHB)%20Model&text=In%20return%20attribution%2C%20allocation%20refers,weights%20in%20the%20bench%2D%20mark.)
- [X] Maximum drawdown

# Currently Supported Feature

- [X] Real-time streaming of PnL, Compound return and Maximum drawdown
- [X] Interactive plot and table
- [X] Schedule to update stock price
- [X] User Login
- [X] Customizable layout

# Tech Stack

[panel](https://panel.holoviz.org/)

# Development

```
git pull https://github.com/Lamonkey/portfolio-monitoring-app.git
cd portfolio-monitoring-app
python -m venv venv
source venvbin/activate
pip install -r requirements.txt
```

Then create a .env file at the project root

```
JQDATA_USER=USERNAME
JQDATA_PASSWORD=PASSWORD
```

Then run the app

```
cd src
panel indexPage.py editingPage.py --setup backgroundTask.py
```
