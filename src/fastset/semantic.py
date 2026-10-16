"""Stable source fingerprints across the supported Python syntax readers."""

import ast
import hashlib
import json


def normalized_ast(source):
    tree = ast.parse(source)
    # Python 3.12 adds empty type_params fields to ordinary classes/functions.
    # Their absence in Python 3.10 is not a model-semantic difference.
    for node in ast.walk(tree):
        if hasattr(node, "type_params") and not node.type_params:
            del node.type_params
    return tree


def fingerprint(source):
    # ast.dump's display of empty fields changed between Python versions.
    # Serialize explicitly rather than fingerprinting that display format.
    def encode(node):
        if isinstance(node, ast.AST):
            return {
                "node": type(node).__name__,
                "fields": {
                    key: encode(value)
                    for key, value in ast.iter_fields(node)
                    if not (key == "type_params" and not value)
                },
            }
        if isinstance(node, list):
            return [encode(value) for value in node]
        return node

    canonical = json.dumps(encode(normalized_ast(source)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
