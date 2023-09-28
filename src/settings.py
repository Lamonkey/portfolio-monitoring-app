from datetime import timedelta
stream_frequency = timedelta(seconds=60)


FREQUENCY = timedelta(hours=24)

TABLE_NAME_AND_FREQ = [
    ('benchmark_profile', timedelta(days=1)),
    ('portfolio_profile', timedelta(days=1))
]

COMPONENT_WIDTH = 375

HANDLE_FEE = 1/1000