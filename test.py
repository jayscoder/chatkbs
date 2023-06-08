import utils

if __name__ == '__main__':
    # history = []
    # prompt = '你好啊'
    # # for response, history in chatai.steam_chat(prompt, history):
    #
    # response, history = chatai.chat(prompt, history)
    # print(response)
    # print(history)

    text = utils.advanced_read_text('data/test.txt')
    chunks = utils.text_to_chunks(text)
    print(chunks[0])
