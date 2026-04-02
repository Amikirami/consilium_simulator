import numpy as np
from openai import OpenAI
from typing import List
from scipy.optimize import linear_sum_assignment


class EmbeddingClient:
    def __init__(self):
        self.client = OpenAI()

    def embed_fn(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """
        Zwraca embedding tekstu jako listę liczb.
        """

        response = self.client.embeddings.create(
            model=model,
            input=text
        )

        return response.data[0].embedding

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Liczy kosinusową podobieństwo między dwoma wektorami.
        """
        num = np.dot(vec1, vec2)
        den = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(num / den)

    # old
    def list_embedding_similarity(self, list1: List[str], list2: List[str], embed_fn) -> float:
        """
        Liczy podobieństwo kosinusowe między embeddingami dwóch list tekstów.
        Embedding listy = średnia embeddingów elementów.
        """
        # embeddingi listy 1
        emb1 = np.array([embed_fn(t) for t in list1])
        mean1 = emb1.mean(axis=0)

        # embeddingi listy 2
        emb2 = np.array([embed_fn(t) for t in list2])
        mean2 = emb2.mean(axis=0)

        # kosinusowa podobieństwo
        return self.cosine_similarity(mean1, mean2)

    def pairwise_similarity_matrix(self, list1: List[str], list2: List[str]) -> np.ndarray:
        """
        Tworzy macierz podobieństw kosinusowych (len(list1) x len(list2)).
        """
        emb1 = [np.array(self.embed_fn(t)) for t in list1]
        emb2 = [np.array(self.embed_fn(t)) for t in list2]

        sim_matrix = np.zeros((len(emb1), len(emb2)))

        for i, e1 in enumerate(emb1):
            for j, e2 in enumerate(emb2):
                sim_matrix[i, j] = self.cosine_similarity(e1, e2)

        return sim_matrix

    def lists_similarity(self, list1: List[str], list2: List[str]) -> float:
        """
        Liczy podobieństwo dwóch list stringów:
        - pairwise similarity
        - Hungarian algorithm
        - zwraca średnie podobieństwo dopasowanych par
        """
        if len(list1) == 0 or len(list2) == 0:
            return 0.0

        sim_matrix = self.pairwise_similarity_matrix(list1, list2)

        # Hungarian algorithm działa na kosztach, więc robimy cost = 1 - similarity
        cost_matrix = 1 - sim_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_similarities = sim_matrix[row_ind, col_ind]

        return float(np.mean(matched_similarities))


