import xapian
import json


class XapianRetriever:
    def __init__(self, db_path="wikipedia_xapian_db", use_bm25=True):
        self.db_path = db_path
        self.db = xapian.Database(db_path)
        self.parser = xapian.QueryParser()
        self.parser.set_stemmer(xapian.Stem("en"))
        self.parser.set_database(self.db)
        self.parser.set_stemming_strategy(xapian.QueryParser.STEM_SOME)
        self.use_bm25 = use_bm25

    def search(self, query, k=10):
        parsed_query = self.parser.parse_query(query)
        enquire = xapian.Enquire(self.db)
        enquire.set_query(parsed_query)

        if self.use_bm25:
            # ✅ BM25Weight with all required positional arguments
            bm25_weight = xapian.BM25Weight()
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
