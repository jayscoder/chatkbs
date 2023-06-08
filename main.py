import gradio as gr
import config
import utils
import json
import kbs
import os


def add_kbs_text(text):
    text_embedding = utils.text_embedding(text)
    md5 = utils.calculate_md5(text)
    chunks = utils.text_to_chunks(text)
    info = {
        'chunks'      : chunks,
        'chunks_count': len(chunks),
        'text_length' : len(text),
        'embedding'   : text_embedding,
        'md5'         : md5
    }
    return json.dumps(info, ensure_ascii=False, indent=2)


def add_kbs_files(files):
    print(files)
    for file in files:
        print(file.name)
    return '上传成功'


def reset_state(count: int):
    if count == 1:
        def reset():
            return []

        return reset
    if count == 2:
        def reset():
            return [], []

        return reset

    if count == 3:
        def reset():
            return [], [], []

        return reset

    def reset():
        return []

    return reset


def reset_user_input():
    return gr.update(value='')


def build_search():
    with gr.Tab('知识库检索'):
        filename_fuzzy_match = gr.Textbox(show_label=False, placeholder='文件名模糊搜索...', lines=1).style(
                container=False)

        search_chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    search_input = gr.Textbox(
                            show_label=False,
                            placeholder="搜索...", lines=10).style(
                            container=False)
                with gr.Column(min_width=32, scale=1):
                    submit_button = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                clear_button = gr.Button("Clear History")
                file_limit = gr.Slider(0, 100, value=3, step=1.0, label="Search File Limit",
                                       interactive=True)
                chunk_limit = gr.Slider(0, 100, value=3, step=1.0, label="Search Chunk Limit",
                                        interactive=True)

                glm_max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length",
                                           interactive=True)
                glm_top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                glm_temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

        submit_button.click(kbs.search_kbs,
                            inputs=[filename_fuzzy_match,
                                    search_input,
                                    search_chatbot,
                                    file_limit,
                                    chunk_limit,
                                    glm_max_length,
                                    glm_top_p,
                                    glm_temperature],
                            outputs=[search_chatbot],
                            show_progress=True)

        clear_button.click(reset_state(1), outputs=[search_chatbot], show_progress=True)


def build_generate():
    with gr.Tab('知识库生成'):
        with gr.Row():
            with gr.Column(scale=4):
                generate_kbs_text_output = gr.Textbox(label='Output')
                with gr.Column(scale=4):
                    update_button = gr.Button('更新')

                with gr.Column(scale=4):
                    reset_button = gr.Button('重置')

            with gr.Column(scale=1):
                chunk_limit = gr.Slider(10, 4096,
                                        value=400,
                                        step=1.0,
                                        label="切块尺寸",
                                        interactive=True)
        update_button.click(kbs.generate_kbs, inputs=[chunk_limit], outputs=generate_kbs_text_output,
                            show_progress=True)
        reset_button.click(kbs.rebuild_kbs, inputs=[chunk_limit], outputs=generate_kbs_text_output,
                           show_progress=True)


def build_chatglm():
    with gr.Tab("ChatGLM-6B"):
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                            container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])

        submitBtn.click(kbs.glm_predict, [user_input, chatbot, max_length, top_p, temperature, history],
                        [chatbot, history],
                        show_progress=True)

        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state(2), outputs=[chatbot, history], show_progress=True)


def build_calculate_embedding():
    import numpy as np
    def chat_embedding(text1, text2):
        em1 = utils.text_embedding(text1)
        em2 = utils.text_embedding(text2)
        em1_np = np.array(em1)
        em2_np = np.array(em2)
        l2_distance = np.linalg.norm(em1_np - em2_np)
        cosine_distance = 1 - np.dot(em1_np, em2_np) / (np.linalg.norm(em1_np) * np.linalg.norm(em2_np))

        return str(em1), str(em2), f'L2距离: {str(l2_distance)}\n余弦距离: {cosine_distance}'

    with gr.Tab("计算Text Embedding"):
        gr.Markdown(
                "[https://tfhub.dev/google/universal-sentence-encoder/4](https://tfhub.dev/google/universal-sentence-encoder/4)")
        with gr.Row():
            with gr.Column(scale=4):
                input_1 = gr.Textbox(label='输入文本')
            with gr.Column(scale=4):
                input_2 = gr.Textbox(label='输入文本')
        with gr.Row():
            with gr.Column(scale=4):
                output_1 = gr.Textbox(label='输出1')
            with gr.Column(scale=4):
                output_2 = gr.Textbox(label='输出2')
        output_distance = gr.Textbox(label='距离')

        button = gr.Button("Calculate")
        button.click(chat_embedding, inputs=[input_1, input_2], outputs=[output_1, output_2, output_distance],
                     show_progress=True)


def pdf2text(files, chatbot: list[tuple[str, str]]):
    for file in files:
        chatbot.append((os.path.basename(file.name), utils.show_text(utils.advanced_read_text(file.name))))
        yield chatbot


def build_pdf2text():
    with gr.Tab("PDF To Text"):
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    files_input = gr.Files(label='Upload your PDF/Txt/Markdown here',
                                           file_types=['.pdf'])
                with gr.Column(min_width=32, scale=1):
                    submit_button = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                clear_button = gr.Button("Clear History")

        submit_button.click(pdf2text, inputs=[files_input, chatbot], outputs=chatbot, show_progress=True)
        clear_button.click(reset_state(1), outputs=[chatbot], show_progress=True)


with gr.Blocks(title='ChatKBS') as demo:
    gr.HTML(f"""<h1 align="center">ChatKBS</h1>""")
    # with gr.Tab('知识库问答'):
    #     pass

    build_search()
    build_generate()
    build_chatglm()
    build_calculate_embedding()
    build_pdf2text()
    # with gr.Tab('输入知识'):
    #     add_kbs_text_input = gr.Textbox(label='输入知识')
    #     add_kbs_text_output = gr.Textbox(label='结果')
    #     add_kbs_text_button = gr.Button("Add")
    #     add_kbs_text_button.click(add_kbs_text, inputs=[add_kbs_text_input], outputs=add_kbs_text_output)
    #
    # with gr.Tab('上传知识'):
    #     add_kbs_files_input = gr.Files(label='Upload your PDF/Txt/Markdown here', file_types=['.pdf', '.txt', '.md'])
    #     add_kbs_files_output = gr.Textbox(label='结果')
    #     add_kbs_files_button = gr.Button("Add")
    #     add_kbs_files_button.click(add_kbs_files, inputs=[add_kbs_files_input], outputs=add_kbs_files_output)

app, _, _ = demo.queue().launch(
        inbrowser=False,
        share=config.SHARE,
        auth=(config.USERNAME, config.PASSWORD),
        server_name=config.SERVER_NAME,
        server_port=config.SERVER_PORT,
        favicon_path=config.FAVICON,
        debug=False
)
