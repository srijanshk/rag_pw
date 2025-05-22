import xapian
import json


class XapianRetriever:
    def __init__(
        self,
        db_path="wikipedia_xapian_db",
        use_bm25=True,
        bm25_k1: float = 1.0,
        bm25_b:  float = 0.25,
    ):
        """
        db_path   : path to the Xapian database
        use_bm25  : whether to use BM25 (else TF-IDF)
        bm25_k1   : BM25 k1 parameter (term-frequency saturation)
        bm25_b    : BM25 b parameter (document-length normalization)
        """
        self.db_path     = db_path
        self.db          = xapian.Database(db_path)
        self.parser      = xapian.QueryParser()
        self.parser.set_stemmer(xapian.Stem("en"))
        self.parser.set_database(self.db)
        self.parser.set_stemming_strategy(xapian.QueryParser.STEM_SOME)
        self.use_bm25     = use_bm25
        self.bm25_k1      = bm25_k1
        self.bm25_b       = bm25_b

    def search(self, query, k=10):
        parsed_query = self.parser.parse_query(query)
        enquire = xapian.Enquire(self.db)
        enquire.set_query(parsed_query)

        if self.use_bm25:
            # BM25Weight(k1, k2, k3, b, min_normlen)
            bm25_weight = xapian.BM25Weight(
                self.bm25_k1,
                0.0,   # k2: disable length-query correction
                1.0,   # k3: within-query-frequency weighting
                self.bm25_b,
                0.0    # min_normlen: minimum normalized document length
            )
            enquire.set_weighting_scheme(bm25_weight)
        else:
            # Optional: fallback to TF-IDF if needed
            enquire.set_weighting_scheme(xapian.TfIdfWeight())

        matches = enquire.get_mset(0, k)
        results = []

        for match in matches:
            doc = match.document
            try:
                metadata = json.loads(doc.get_data())
                results.append((metadata, match.percent / 100.0))  # normalized score
            except Exception as e:
                print(f"[XapianRetriever] ⚠️ Failed to decode document: {e}")
                continue

        return results

    def set_bm25_parameters(self, k1: float, b: float):
        """
        Update BM25 parameters for subsequent searches.
          k1 controls term-frequency saturation (e.g. 1.0–2.0)
          b  controls length normalization (e.g. 0.0–1.0)
        """
        self.bm25_k1 = k1
        self.bm25_b  = b
