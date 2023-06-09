from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

import config

_DIMENSION = 512

_NLIST = _DIMENSION * 40

_NPROB = int(_NLIST / 15)

connections.connect("default", host="localhost", port="19530")

_kbs_file_schema = CollectionSchema([
    FieldSchema(name="filename_md5", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=32),
    FieldSchema(name="text_md5", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_DIMENSION)
], "")

_kbs_chunk_schema = CollectionSchema([
    FieldSchema(name="filename_md5_no", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=50),
    FieldSchema(name="filename_md5", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="chunk_no", dtype=DataType.INT64),
    FieldSchema(name="chunk_md5", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_DIMENSION)
], "")

kbs_file_milvus = Collection(config.MILVUS_COLLECTION_KBS_FILE, _kbs_file_schema)
kbs_chunk_milvus = Collection(config.MILVUS_COLLECTION_KBS_CHUNK, _kbs_chunk_schema)

_l2_index = {
    "index_type" : "IVF_FLAT",
    "metric_type": "L2",
    "params"     : { "nlist": _NLIST },
}
_ip_index = {
    "index_type" : "IVF_FLAT",
    "metric_type": "IP",
    "params"     : { "nlist": _NLIST },
}


def rebuild():
    global kbs_file_milvus, kbs_chunk_milvus
    kbs_file_milvus.drop()
    kbs_chunk_milvus.drop()

    kbs_file_milvus = Collection(config.MILVUS_COLLECTION_KBS_FILE, _kbs_file_schema)
    kbs_chunk_milvus = Collection(config.MILVUS_COLLECTION_KBS_CHUNK, _kbs_chunk_schema)

    kbs_file_milvus.create_index("embedding", _l2_index)
    kbs_chunk_milvus.create_index("embedding", _l2_index)


kbs_file_milvus.create_index("embedding", _l2_index)
kbs_chunk_milvus.create_index("embedding", _l2_index)

kbs_file_milvus.load()
kbs_chunk_milvus.load()


def file_insert(filename_md5: str, text_md5: str, embedding: list[float]):
    kbs_file_milvus.insert([[filename_md5], [text_md5], [embedding]])


def chunk_insert(filename_md5, chunk_no: int, chunk_md5: str, embedding: list[float]):
    kbs_chunk_milvus.insert([[f'{filename_md5}_{chunk_no}'], [filename_md5], [chunk_no], [chunk_md5], [embedding]])


def file_search(embedding: list[float], limit: int, metric_type: str = 'L2') -> list[str]:
    search_params = { "metric_type": metric_type, "params": { "nprobe": _NPROB } }

    results = kbs_file_milvus.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=None,
            consistency_level="Strong",
            output_fields=['filename_md5']
    )

    filename_md5_list = []
    for hits in results:
        for hit in hits:
            filename_md5 = hit.entity.get('filename_md5')
            filename_md5_list.append((filename_md5, hit.distance))

    print(filename_md5_list)
    filename_md5_list = sorted(filename_md5_list, key=lambda x: x[1])
    filename_md5_list = list(map(lambda x: x[0], filename_md5_list))

    return filename_md5_list


def chunk_search(embedding: list[float], limit: int, metric_type: str = 'L2') -> list[str]:
    search_params = { "metric_type": metric_type, "params": { "nprobe": _NPROB } }

    results = kbs_chunk_milvus.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=['filename_md5_no']
    )

    filename_md5_no_list = []

    for hits in results:
        for hit in hits:
            filename_md5_no = hit.entity.get('filename_md5_no')
            filename_md5_no_list.append((filename_md5_no, hit.distance))

    filename_md5_no_list = sorted(filename_md5_no_list, key=lambda x: x[1])
    filename_md5_no_list = list(map(lambda x: x[0], filename_md5_no_list))

    return filename_md5_no_list
