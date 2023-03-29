import os

redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_get_request_socket_timeout = 20 # s
redis_resp_expire = 60*60*2  # 2h
