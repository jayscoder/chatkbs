import hashlib
import nltk
import os
import fitz
import re
from bs4 import BeautifulSoup
import unicodedata

_EMBED = None

nltk.download('punkt')
nltk.download('stopwords')


# 将全角字符替换为半角字符
def replace_fullwidth_chars(text):
    # 创建一个空字符串来存储替换后的文本
    replaced_text = ""
    pattern = '[\u3000]+'
    text = re.sub(pattern, ' ', text)
    # 遍历字符串中的每个字符
    for char in text:
        # 获取字符的Unicode名称
        name = unicodedata.name(char, "")

        # 检查字符是否是全角字符
        if "FULLWIDTH" in name:
            # 获取全角字符的对应半角字符
            halfwidth_char = unicodedata.normalize('NFKC', char)
            replaced_text += halfwidth_char
        else:
            # 如果字符不是全角字符，则直接添加到替换后的文本中
            replaced_text += char

    return replaced_text


def calculate_md5(text: str) -> str:
    md5_hash = hashlib.md5()
    md5_hash.update(text.encode())
    digest = md5_hash.hexdigest()
    return digest


def text_to_tokens(text: str) -> list[str]:
    import jieba
    return list(jieba.cut(text, cut_all=True))


def text_to_sentences(text: str):
    sentences = []
    current_sent = ''
    text_length = len(text)

    idx = 0
    while idx < text_length:
        ch = text[idx]
        if idx == 0 or current_sent == '':
            current_sent = ch
        elif idx == text_length - 1:
            current_sent += ch
        else:
            current_sent += ch
            if ch in '.。!！?？;；':
                current_sent = current_sent.strip()
                if current_sent != '':
                    sentences.append(current_sent)
                current_sent = ''

        idx += 1
    current_sent = current_sent.strip()
    if current_sent != '':
        sentences.append(current_sent)
    return sentences


def text_to_chunks(text: str, size: int = 400, overlap: int = 0, limit: int = 100):
    parts = []
    for line in text.split('\n'):
        line = line.strip()
        if line == '':
            continue
        if len(line) < size / 10:
            parts.append(line)
            continue

        sentences = text_to_sentences(line)

        for sent in sentences:
            if len(sent) < size / 10:
                parts.append(sent)
                continue

            for word in nltk.word_tokenize(sent):
                parts.append(word)

    chunks = []

    current_chunks = []
    current_chunk_length = 0

    for idx, part in enumerate(parts):
        current_chunks.append(part)
        current_chunk_length += len(part)
        if current_chunk_length > size:
            chunk = ' '.join(current_chunks).strip()
            if len(chunk) > 0:
                chunks.append(chunk)
                if len(chunks) >= limit:
                    return chunks

            current_chunks = []
            current_chunk_length = 0

            for oi in range(idx - overlap + 1, idx + 1):
                current_chunks.append(parts[oi])
                current_chunk_length += len(parts[oi])

    if current_chunk_length > 0:
        chunk = ' '.join(current_chunks).strip()
        if len(chunk) > 0:
            chunks.append(chunk)

    return chunks


def pdf_to_text(path: str):
    doc = fitz.open(path)
    text_list = []

    for i in range(0, doc.page_count):
        text = doc.load_page(i).get_text("text")
        text_list.append(text)

    doc.close()
    return '\n'.join(text_list)


def html_to_text(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    return text


def text_preprocess(text: str):
    text = replace_fullwidth_chars(text)
    text = re.sub('\n{2,}', '\n', text)
    text = re.sub(' {2,}', ' ', text)
    return text


# 高级读取文本方案
def advanced_read_text(filepath: str) -> str:
    _, ext = os.path.splitext(filepath)
    if ext == '.pdf':
        file_text = pdf_to_text(filepath)
    elif ext == '.html':
        file_text = html_to_text(filepath)
    elif ext == '.caj':
        file_text = ''
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                file_text = f.read()
            except:
                file_text = ''
    file_text = text_preprocess(file_text)
    return file_text


def list_files(dir_path: str) -> [str]:
    filenames = set()

    for root, dirs, files in os.walk(dir_path):
        # 遍历目录下的文件
        for file in files:
            filepath = os.path.join(root, file)
            name, ext = os.path.splitext(file)
            if ext in ['.caj']:
                # 将caj处理成pdf然后再读取
                pdf_filename = f'{name}.pdf'
                pdf_filepath = os.path.join(root, pdf_filename)
                if os.path.exists(pdf_filepath):
                    filenames.add(pdf_filename)
                    continue
                try:
                    os.system(f'./caj2pdf/caj2pdf convert {filepath} -o {pdf_filepath}')
                    if os.path.exists(pdf_filepath):
                        filenames.add(pdf_filename)
                except:
                    continue
            else:
                filenames.add(file)
    return list(filenames)


def show_text(text):
    if text == '':
        return ''

    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text
