import gradio as gr
import config
import tensorflow_hub as hub

embed = hub.load("https://wangtong15.oss-cn-beijing.aliyuncs.com/models/universal-sentence-encoder_4.tar.gz")

def chat_embedding(text):
    embeddings = embed([text])

    print(embeddings)
    return str(embeddings[0].numpy().tolist())

with gr.Blocks() as demo:
    with gr.Tab("Flip Text"):
        gr.Markdown("Flip text or image files using this demo.")
        embedding_text_input = gr.Textbox()
        embedding_text_output = gr.Textbox()
        embedding_text_button = gr.Button("Flip")

    embedding_text_button.click(chat_embedding, inputs=embedding_text_input, outputs=embedding_text_output)


app, _, _ = demo.launch(
        inbrowser=False,
        share=False,
        auth=(config.USERNAME, config.PASSWORD),
        server_name='0.0.0.0',
        server_port=10001,
        favicon_path='favicon.png',
)

app.title = 'ChatKBS'
