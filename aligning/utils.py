from typing import Set, List, Union, Tuple
from Bio import Align

ALPHABET_ASCII = {chr(i) for i in range(128)}
ALPHABET_LATIN_1 = {chr(i) for i in range(256)}

class Word:
    def __init__(self, word: str, start_time: float, end_time: float, confidence: float):
        self.word = word
        self.start_time = start_time
        self.end_time = end_time
        self.confidence = confidence

    def __repr__(self):
        return f'Word(word={self.word}, start_time={self.start_time}, end_time={self.end_time}, confidence={self.confidence})'


def create_aligner_global(
    alphabet: Set[str],
    match_score: float,
    mismatch_score: float,
    target_left_open_gap_score: float, target_internal_open_gap_score: float, target_right_open_gap_score: float,
    target_left_extend_gap_score: float, target_internal_extend_gap_score: float, target_right_extend_gap_score: float,
    query_left_open_gap_score: float, query_internal_open_gap_score: float, query_right_open_gap_score: float,
    query_left_extend_gap_score: float, query_internal_extend_gap_score: float, query_right_extend_gap_score: float
) -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.alphabet = sorted(alphabet)
    aligner.mode = 'global'

    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.target_left_open_gap_score = target_left_open_gap_score
    aligner.target_internal_open_gap_score = target_internal_open_gap_score
    aligner.target_right_open_gap_score = target_right_open_gap_score
    aligner.target_left_extend_gap_score = target_left_extend_gap_score
    aligner.target_internal_extend_gap_score = target_internal_extend_gap_score
    aligner.target_right_extend_gap_score = target_right_extend_gap_score
    aligner.query_left_open_gap_score = query_left_open_gap_score
    aligner.query_internal_open_gap_score = query_internal_open_gap_score
    aligner.query_right_open_gap_score = query_right_open_gap_score
    aligner.query_left_extend_gap_score = query_left_extend_gap_score
    aligner.query_internal_extend_gap_score = query_internal_extend_gap_score
    aligner.query_right_extend_gap_score = query_right_extend_gap_score

    return aligner



class AlignedSentence:
    def __init__(self, sentence: str, start_time: Union[float, None], end_time: Union[float, None]):
        self.sentence = sentence
        self.start_time = start_time
        self.end_time = end_time


def data_to_data_sentence_only(
    data: List[Tuple[List[AlignedSentence], List[Word]]]
) -> List[Tuple[List[str], List[Word]]]:
    return [
        (
            [aligned_sentence.sentence for aligned_sentence in sentence_alignment],
            stt_output
        )
        for sentence_alignment, stt_output in data
    ]

def preprocess_transcript_for_alignment(transcript: str, alphabet: Set[str]):
    transcript = ''.join([char for char in transcript if char in alphabet])
    transcript = transcript.strip()

    return transcript