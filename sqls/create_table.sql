CREATE TABLE IF NOT EXISTS "kbs_file"
(
    "id"           INTEGER PRIMARY KEY,
    "filename_md5" TEXT UNIQUE,
    "text"         TEXT,
    "filename"     TEXT,
    "summary"      TEXT,
    "type"         TEXT,
    "create_time"  TIMESTAMP,
    "text_length"  INTEGER,
    "embedding"    TEXT,
    "text_md5"     TEXT,
    "update_time"  TIMESTAMP
);
CREATE TABLE IF NOT EXISTS "kbs_chunk"
(
    "id"              INTEGER PRIMARY KEY,
    "filename_md5"    TEXT,
    "filename_md5_no" TEXT UNIQUE,
    "chunk"           TEXT,
    "chunk_no"        INTEGER,
    "chunk_length"    INTEGER,
    "summary"         TEXT,
    "filename"        TEXT,
    "create_time"     TIMESTAMP,
    "embedding"       TEXT,
    "chunk_md5"       TEXT,
    "update_time"     TIMESTAMP
);
