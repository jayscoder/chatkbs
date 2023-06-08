import os

VERSION_NAME = '1.0.3'
VERSION_CODE = 4

# 获取当前工作目录的绝对路径
BASE_DIR = os.getcwd()

USERNAME = 'chatkbs'
PASSWORD = 'chatkbs'
SERVER_PORT = 10001
SERVER_NAME = '0.0.0.0'
FAVICON = 'favicon.png'
SHARE = False

DATA_DIR = 'data'

SQLITE_DATABASE = f'database-{VERSION_NAME}.db'

CHATGLM_MODEL_PATH = 'THUDM/chatglm-6b'

MILVUS_COLLECTION_KBS_CHUNK = f'kbs_chunk_{VERSION_CODE}'
MILVUS_COLLECTION_KBS_FILE = f'kbs_file_{VERSION_CODE}'
