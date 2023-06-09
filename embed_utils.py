import tensorflow_hub as hub

embed = hub.load("https://wangtong15.oss-cn-beijing.aliyuncs.com/models/universal-sentence-encoder_4.tar.gz")

def calculate_embedding(text: str) -> list[float]:
    return embed([text]).numpy().tolist()[0]
