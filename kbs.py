from typing import Union
import os
import utils
import config
import sqlite3
import time
from datetime import datetime
import json
from collections import defaultdict
import db_utils
import db_milvus
import chatai


def rebuild_kbs(chunk_size: int, chunk_overlap: int, chunk_limit: int):
    db_utils.rebuild()
    yield from generate_kbs(chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunk_limit=chunk_limit)


def generate_kbs(chunk_size: int, chunk_overlap: int, chunk_limit: int):
    new_filenames = utils.list_files(config.DATA_DIR)

    outputs = f'{config.DATA_DIR}中共有{len(new_filenames)}个文件: '

    yield outputs

    # 查询 "filename" 和 "md5" 列的数据
    with sqlite3.connect(config.SQLITE_DATABASE) as conn:
        cursor = conn.cursor()

        # 获取查询结果
        results = cursor.execute('''
    SELECT filename, text_md5
    FROM kbs_file
    ''').fetchall()

        old_file_md5 = defaultdict(str)
        old_filenames = []
        for row in results:
            old_file_md5[row[0]] = row[1]
            old_filenames.append(row[0])

        merged_list = list(set(old_filenames + new_filenames))
        total = len(merged_list)

        outputs = [
            f'{outputs}\n知识库中已有{len(results)}个文件\n一共需要比对{total}个文件'
        ]

        for idx, filename in enumerate(merged_list):
            outputs.append(f'[{idx + 1}/{total}] {filename}: 等待处理')

        yield '\n'.join(outputs)

        for idx, filename in enumerate(merged_list):
            for output in generate_kbs_file(root=config.DATA_DIR, filename=filename, old_md5=old_file_md5[filename],
                                            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                            chunk_limit=chunk_limit):
                outputs[idx + 1] = f'[{idx + 1}/{total}] {filename}: {output}'
                yield '\n'.join(outputs)

        outputs.append('全部已经处理完成')
        yield '\n'.join(outputs)


def generate_kbs_file(
        root: str,
        filename: str,
        old_md5: str,
        chunk_size: int,
        chunk_overlap: int,
        chunk_limit: int) -> str:
    import embed_utils

    filepath = os.path.join(root, filename)

    filename_md5 = utils.calculate_md5(filename)

    with sqlite3.connect(config.SQLITE_DATABASE) as conn:
        cursor = conn.cursor()

        if not os.path.exists(filepath):
            db_utils.delete_all_by_filename_md5(cursor=cursor, filename_md5=filename_md5)
            yield '已删除'
            return

        _, ext = os.path.splitext(filename)
        file_raw_text = utils.advanced_read_text(filepath)
        file_full_text = f'文件 {filename}' + '\n' + file_raw_text
        file_text_md5 = utils.calculate_md5(file_full_text)
        file_raw_text_length = len(file_raw_text)

        if file_text_md5 == old_md5:
            yield '无改动'
            return

        yield f'字符数={file_raw_text_length}'

        file_text_embedding = embed_utils.calculate_embedding(file_full_text)

        filetype = ext.replace('.', '')
        # 获取当前的13位时间戳（以毫秒为单位）
        timestamp = int(time.time() * 1000)

        db_utils.delete_all_by_filename_md5(cursor=cursor, filename_md5=filename_md5)

        kbs_file_data = (file_raw_text, filename_md5, filename, '', filetype, timestamp, len(file_raw_text),
                         json.dumps(file_text_embedding, ensure_ascii=False), file_text_md5, timestamp)

        cursor.execute('''INSERT OR REPLACE INTO kbs_file (text, filename_md5, filename, summary, type, create_time, text_length, embedding, text_md5, update_time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', kbs_file_data)

        conn.commit()

        db_milvus.file_insert(
                filename_md5=filename_md5,
                text_md5=file_text_md5,
                embedding=file_text_embedding
        )

        chunks = utils.text_to_chunks(file_raw_text, size=chunk_size, overlap=chunk_overlap, limit=chunk_limit)
        for no, chunk in enumerate(chunks):
            chunk_full_text = f'文件 {filename} 第{no}部分\n' + chunk

            chunk_md5 = utils.calculate_md5(chunk_full_text)
            chunk_embedding = embed_utils.calculate_embedding(chunk_full_text)

            sql = '''INSERT OR REPLACE INTO kbs_chunk (filename_md5, filename_md5_no, chunk, chunk_no, chunk_length, summary, filename, create_time, embedding, chunk_md5, update_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    '''

            filename_md5_no = f'{filename_md5}_{no}'
            kbs_chunk_data = (
                filename_md5,
                filename_md5_no,
                chunk,
                no,
                len(chunk),
                '',
                filename,
                timestamp,
                json.dumps(chunk_embedding, ensure_ascii=False),
                chunk_md5,
                timestamp
            )
            cursor.execute(sql, kbs_chunk_data)
            conn.commit()

            db_milvus.chunk_insert(
                    filename_md5=filename_md5,
                    chunk_no=no,
                    chunk_md5=chunk_md5,
                    embedding=chunk_embedding
            )

            yield f'[{no + 1}/{len(chunks)}] 字符数={file_raw_text_length}'
    db_milvus.kbs_file_milvus.flush()
    db_milvus.kbs_chunk_milvus.flush()
    yield f'👌 字符数={file_raw_text_length}'


def search_kbs(filename_fuzzy_match: str,
               search_input: str,
               chatbot: list[tuple[str, str]],
               search_file_limit: int,
               search_chunk_limit: int, chunk_limit: int,
               search_metric_type: str,
               glm_max_length: int,
               glm_top_p: float,
               glm_temperature: float):
    import embed_utils
    print(
            f'search_file_limit={search_file_limit} search_chunk_limit={search_chunk_limit} search_metric_type={search_metric_type}')
    chatbot[:] = []
    chatbot.append((utils.show_text(search_input), ""))

    search_embedding = embed_utils.calculate_embedding(search_input)
    db_milvus.kbs_chunk_milvus.load()
    db_milvus.kbs_file_milvus.load()

    with sqlite3.connect(config.SQLITE_DATABASE) as conn:
        cursor = conn.cursor()

        if search_file_limit > 0:
            # 先查询可能的文件
            filename_md5_list = db_milvus.file_search(
                    embedding=search_embedding,
                    limit=search_file_limit,
                    metric_type=search_metric_type)

            # 构建逗号分隔的参数字符串
            results = cursor.execute(
                    f'SELECT filename FROM kbs_file WHERE filename_md5 IN ({db_utils.que_marks(len(filename_md5_list))})',
                    filename_md5_list).fetchall()

            filenames = [row[0] for row in results]

            chatbot.append(("你可能感兴趣的文件", utils.show_text('\n'.join(filenames))))

            yield chatbot

        if search_chunk_limit > 0:
            # 搜索chunk
            filename_md5_no_list = db_milvus.chunk_search(
                    embedding=search_embedding,
                    limit=search_chunk_limit,
                    metric_type=search_metric_type)
            results = cursor.execute(
                    f'SELECT filename, chunk_no, chunk FROM kbs_chunk WHERE kbs_chunk.filename_md5_no IN ({db_utils.que_marks(len(filename_md5_no_list))})',
                    filename_md5_no_list).fetchall()

            for row in results:
                chatbot.append((utils.show_text(f'文件 {row[0]} 第{row[1]}部分'), utils.show_text(row[2])))
                yield chatbot

    # for response, history in chatai.stream_chat(search_input, None, max_length=glm_max_length, top_p=glm_top_p,
    #                                             temperature=glm_temperature):
    #     chatbot[-1] = (utils.show_text(input), utils.show_text(response))
    #
    #     yield chatbot


def glm_predict(input_text, chatbot, max_length, top_p, temperature, history):
    chatbot.append((utils.show_text(input_text), ""))
    yield chatbot, history
    for response, history in chatai.stream_chat(
            input_text,
            history=history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature):
        chatbot[-1] = (utils.show_text(input_text), utils.show_text(response))
        yield chatbot, history


def files_long_predict(
        files,
        input_text,
        chatbot,
        read_mode: str,
        chunk_size: int,
        chunk_overlap: int,
        chunk_limit: int,
        repeat: int,
        max_length,
        top_p,
        temperature,
        history,
):
    chatbot.append((utils.show_text(input_text), ""))
    yield chatbot, history
    total_chunks = []

    for file in files:
        text = utils.advanced_read_text(file.name)
        chunks = utils.text_to_chunks(text, size=chunk_size, overlap=chunk_overlap, limit=chunk_limit)
        filename = os.path.basename(file.name)
        for chunk_no, chunk in enumerate(chunks):
            total_chunks.append((filename, chunk_no, chunk))

    if read_mode == 'Recursive':
        total = len(total_chunks) * repeat
        progress_i = 0
        memory = ''
        for rpi in range(repeat):
            for (filename, chunk_no, chunk) in total_chunks:
                prompt = f'当前上下文:\n{memory}\n---\n新的上下文片段:\n文件名={filename} 片段序号={chunk_no}\n{chunk}\n---\n根据用户关心的问题\"{input_text}\"，结合新的上下文片段，生成新的上下文：'
                cache_memory = ''
                for response, history in chatai.stream_chat(
                        prompt,
                        history=history,
                        max_length=max_length,
                        top_p=top_p,
                        temperature=temperature):
                    progress_text = f'第{rpi + 1}次阅读: [{filename} 第{chunk_no + 1}部分] 进度: {progress_i}/{total}'
                    chatbot[-1] = (utils.show_text(input_text) + f"\n---\n{memory}", utils.show_text(
                            f"{progress_text}\n{response}"))
                    # 显示文本
                    yield chatbot, history
                    cache_memory = response
                memory = cache_memory

                # 丢弃history的最后一项
                history = history[:-1]
                yield chatbot, history
                progress_i += 1

        prompt = f'上下文: {memory}\n---\n{input_text}'

        for response, history in chatai.stream_chat(
                prompt,
                history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature
        ):
            chatbot[-1] = (utils.show_text(input_text), utils.show_text(response))
            yield chatbot, history
    else:
        # ForEach
        total = len(total_chunks) * repeat
        progress_i = 0

        total_history = [[] for i in range(len(total_chunks))]
        total_results = ['' for i in range(len(total_chunks))]

        for rpi in range(repeat):
            for idx, (filename, chunk_no, chunk) in enumerate(total_chunks):
                history = total_history[idx]
                if len(history) > 0:
                    prompt = f'上下文片段:\n文件名={filename}\n{chunk}\n---\n\"{input_text}'
                else:
                    prompt = f'上下文片段:\n文件名={filename}\n{chunk}\n---\n(优化上次的回答内容)\"{input_text}'
                cache_response = ''
                for response, history in chatai.stream_chat(
                        prompt,
                        history=history,
                        max_length=max_length,
                        top_p=top_p,
                        temperature=temperature):
                    progress_text = f'第{rpi + 1}次阅读: [{filename} 第{idx + 1}部分] 进度: {progress_i}/{total}'
                    show_results = '\n\n'.join(total_results)

                    chatbot[-1] = (utils.show_text(input_text) + f"\n\n{show_results}", utils.show_text(
                            f"{progress_text}\n{response}"))
                    # 显示文本
                    yield chatbot, history
                    cache_response = ''

                total_results[idx] = cache_response
                total_history[idx] = history
                progress_i += 1


def text_long_predict(
        context_text,
        input_text,
        chatbot,
        read_mode: str,
        chunk_size: int,
        chunk_overlap: int,
        chunk_limit: int,
        repeat: int,
        max_length,
        top_p,
        temperature,
        history,
):
    chatbot.append((utils.show_text(input_text), ""))
    yield chatbot, history
    chunks = utils.text_to_chunks(context_text, size=chunk_size, overlap=chunk_overlap, limit=chunk_limit)

    total = len(chunks) * repeat
    memory = ''
    progress_i = 0
    if total > 1:
        for rpi in range(repeat):
            for idx, chunk in enumerate(chunks):
                prompt = f'当前上下文:\n{memory}\n---\n新的上下文片段:\n{chunk}\n---\n根据用户关心的问题\"{input_text}\"，结合新的上下文片段，生成新的上下文：'
                cache_memory = ''
                for response, history in chatai.stream_chat(
                        prompt,
                        history=history,
                        max_length=max_length,
                        top_p=top_p,
                        temperature=temperature):
                    progress_text = f'第{rpi + 1}次阅读: [第{idx + 1}部分] 进度: {progress_i}/{total}'
                    chatbot[-1] = (utils.show_text(input_text) + f"\n---\n{memory}", utils.show_text(
                            f"{progress_text}\n{response}"))
                    # 显示文本
                    yield chatbot, history
                    cache_memory = response
                memory = cache_memory

                # 丢弃history的最后一项
                history = history[:-1]
                yield chatbot, history
                progress_i += 1
    elif len(chunks) > 0:
        memory = chunks[0]

    prompt = f'上下文: {memory}\n---\n{input_text}'

    for response, history in chatai.stream_chat(
            prompt,
            history=history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature
    ):
        chatbot[-1] = (utils.show_text(input_text), utils.show_text(response))
        yield chatbot, history
