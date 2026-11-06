from transformers import AutoProcessor, WhisperModel

class Whisper():
    def __init__(self):
        self.model_name = "openai/whisper-large-v3-turbo"
        self.__whisper_model = WhisperModel.from_pretrained(self.model_name, device_map="auto", attn_implementation="sdpa")
        self.__processor = AutoProcessor.from_pretrained(self.model_name)

    @property
    def whisper_model(self):
        return self.__whisper_model

    @property
    def processor(self):
        return self.__processor
    