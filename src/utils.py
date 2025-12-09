import re
from collections.abc import Iterable


def normalize_sparql(query: str) -> list:
    """
    Normalizes a SPARQL query:
    - remove trailing and leading whitespaces
    - convert to lowercase
    - remove alias patterns
    - canonicalize variables
    """
    query_cleaned = query.strip().lower()
    query_cleaned = remove_aliases(query_cleaned)
    query_cleaned = canonicalize_variables(query_cleaned)

    return query_cleaned


def canonicalize_variables(query: str) -> str:
    """
    Replaces all SPARQL variables (e.g., ?x, ?author) 
    with canonical names (?v1, ?v2, ...).
    """
    var_pattern = re.compile(r'\?[\w_]+')
    seen_vars = {}
    var_counter = 1

    def replace_var(match):
        nonlocal var_counter
        var_name = match.group()
        if var_name not in seen_vars:
            seen_vars[var_name] = f"?v{var_counter}"
            var_counter += 1
        return seen_vars[var_name]

    return var_pattern.sub(replace_var, query)


def tokenize_sparql(query: str) -> list:
    """
    Tokenizes a SPARQL query into a list of meaningful tokens.
    - separates punctuation: { } ( ) . ; , = etc.
    - keeps URIs <...> and string literals "...", '...'
    - keeps variables ?x, ?var, etc.
    """

    # protect string literals and URIs
    uri_pattern = r'<[^>]*>'
    string_pattern = r'"[^"]*"|\'.*?\''
    protected = re.findall(f'{uri_pattern}|{string_pattern}', query)
    placeholders = {f"__PLACEHOLDER_{i}__": val for
                    i, val in enumerate(protected)}
    for i, val in enumerate(protected):
        query = query.replace(val, f"__PLACEHOLDER_{i}__")

    # tokenize punctuation and operators
    query = re.sub(r'([{}()\[\];.,=])', r' \1 ', query)
    query = re.sub(r'\s+', ' ', query).strip()

    tokens = query.split()

    # restore protected values
    tokens = [placeholders.get(tok, tok) for tok in tokens]

    return tokens


def remove_aliases(query: str) -> str:
    """
    Remove redundant aliases, i.e. patterns where a variable is aliased to itself.
    """
    # within parentheses
    query = re.sub(
        r'\(\s*(\?[A-Za-z_]\w*)\s+AS\s+\1\s*\)',
        r'\1',
        query,
        flags=re.IGNORECASE,
    )

    # without parentheses
    query = re.sub(
        r'(\?[A-Za-z_]\w*)\s+AS\s+\1\b',
        r'\1',
        query,
        flags=re.IGNORECASE,
    )

    return query


def normalize_results(results) -> set:
    """
    Turn a result (list/iterable of rows) into a set of frozensets.

    - Each row is an iterable of hashable values (e.g. tuple/list of strings)
    - Order of rows is ignored
    - Order inside rows is ignored
    - Duplicate rows are ignored
    """
    if results is None:
        return set()

    # Accept single row or full list-of-rows
    if not isinstance(results, Iterable) or isinstance(results, (str, bytes)):
        results = [results]

    norm = set()
    for row in results:
        # Treat non-iterable or strings as a single-element row
        if not isinstance(row, Iterable) or isinstance(row, (str, bytes)):
            row = [row]
        # row can be (), ('x',), ('x','y'), ['x',...], ...
        norm.add(frozenset(row))
    return norm


def fbeta_score(generated: set, reference: set, beta=1.0) -> float:
    """
    F-beta between two result sets.
    """
    tp = len(generated & reference)
    fp = len(generated - reference)
    fn = len(reference - generated)

    betasq = beta**2
    denom = (1 + betasq) * tp + fp + betasq * fn
    if denom == 0:
        return 0.0

    return ((1 + betasq) * tp) / denom
