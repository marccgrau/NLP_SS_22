from abc import ABC, abstractmethod
from typing import List

from .utils import Word


class SttOutputReader(ABC):
    @abstractmethod
    def read(self, path_to_stt_output: str) -> List[Word]:
        pass

class NeMoSttOutputReader(SttOutputReader):
    def read(self, path_to_stt_output: str) -> List[Word]:
        with open(path_to_stt_output, encoding='utf-8') as f:
            stt_output_orig = json.load(f)

        stt_output = []
        for word_dict in stt_output_orig['results']['items']:
            if word_dict['type'] != 'punctuation':
                word = word_dict['alternatives'][0]['content']
                start_time = float(word_dict['start_time'])
                end_time = float(word_dict['end_time'])
                stt_output.append(Word(word, start_time, end_time))

        return stt_output