import os

debug=False
if debug == True:
    redis_host='localhost'
else:
    redis_host=os.getenv('REDIS_HOST')
