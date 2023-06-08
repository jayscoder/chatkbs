import sqlite3
import config
import os
import db_milvus


# 连接到数据库（如果数据库不存在，则会创建一个新的数据库）
def execute_sql_file(file_path):
    with sqlite3.connect(config.SQLITE_DATABASE) as conn:
        cursor = conn.cursor()
        with open(file_path, 'r', encoding='utf-8') as f:
            sqls = f.read().split(';')
            for sql in sqls:
                sql = sql.strip()
                if sql == '':
                    continue
                cursor.execute(sql)


def rebuild():
    if os.path.exists(config.SQLITE_DATABASE):
        os.remove(config.SQLITE_DATABASE)

    execute_sql_file('sqls/create_table.sql')

    db_milvus.rebuild()


def delete_all_by_filename_md5(cursor: sqlite3.Cursor, filename_md5: str):
    db_milvus.kbs_file_milvus.delete(f"filename_md5 in ['{filename_md5}']")

    chunk_rows = cursor.execute(f"SELECT filename_md5_no FROM kbs_chunk WHERE filename_md5 = ?",
                                (filename_md5,)).fetchall()
    deleted_filename_md5_no_list = [row[0] for row in chunk_rows]

    db_milvus.kbs_chunk_milvus.delete(f"filename_md5_no in {deleted_filename_md5_no_list}")

    db_milvus.kbs_file_milvus.flush()
    db_milvus.kbs_chunk_milvus.flush()

    # 文件不存在了，直接从sqlite中删除
    cursor.execute("DELETE FROM kbs_file WHERE filename_md5 = ?", (filename_md5,))
    cursor.execute("DELETE FROM kbs_chunk WHERE filename_md5 = ?", (filename_md5,))


def delete_chunk_by_filename_md5(cursor: sqlite3.Cursor, filename_md5: str):
    chunk_rows = cursor.execute(f"SELECT filename_md5_no FROM kbs_chunk WHERE filename_md5 = ?",
                                (filename_md5,)).fetchall()
    deleted_filename_md5_no_list = [row[0] for row in chunk_rows]

    db_milvus.kbs_chunk_milvus.delete(f"filename_md5_no in {deleted_filename_md5_no_list}")
    db_milvus.kbs_chunk_milvus.flush()

    cursor.execute("DELETE FROM kbs_chunk WHERE filename_md5 = ?", (filename_md5,))


def que_marks(count: int) -> str:
    return ', '.join(['?'] * count)


if not os.path.exists(config.SQLITE_DATABASE):
    # sqlite数据库不存在了
    rebuild()
