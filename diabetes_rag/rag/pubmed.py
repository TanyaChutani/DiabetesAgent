from Bio import Entrez

Entrez.email = "your_email@example.com"


def fetch_pubmed_abstracts(query, max_results=2):

    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        ids = record["IdList"]

        if len(ids) == 0:
            return ""

        handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="text", rettype="abstract")
        abstracts = handle.read()

        return abstracts

    except Exception as e:
        print("PubMed error:", e)
        return ""