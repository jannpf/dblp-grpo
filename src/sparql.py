import json

from SPARQLWrapper import SPARQLWrapper, JSON, POST, POSTDIRECTLY
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

from rdflib.plugins.sparql.parser import parseQuery


def execute_sparql(query: str, endpoint) -> dict:
    # sometimes special chars
    if "\\" in query:
        query = query.encode('utf-8').decode('unicode_escape')

    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    # sparql.setMethod(POST)
    # sparql.setRequestMethod(POSTDIRECTLY)
    sparql.setReturnFormat(JSON)

    # sparql.setTimeout(5)

    response: dict = sparql.query().convert()  # type: ignore

    return response


def validate_query(query: str) -> bool:
    try:
        parseQuery(query)
        return True
    except Exception as e:
        return False
