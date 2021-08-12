import os
import json

from itertools import chain
from jsonschema import Draft7Validator, RefResolver, ValidationError
from utils import resolve_relative_path

# "base_path=None": use directory of script as base_path
schema_base_dir = resolve_relative_path('./schema/', base_path=None)
design_pattern_schema_base_dir = resolve_relative_path('./schema/design_patterns/', base_path=None)
method_schema = None
schema_store = {}
schema_resolver = None
schema_validator = None


def init_schemata():
    global method_schema, schema_store
    if not os.path.isdir(schema_base_dir):
        return False, 'Schema directory "%s" does not exist' % schema_base_dir
    if not os.path.isdir(design_pattern_schema_base_dir):
        return False, 'Design pattern schema directory "%s" does not exist' % design_pattern_schema_base_dir

    method_schema_path = os.path.join(schema_base_dir, 'method.json')
    if not os.path.isfile(method_schema_path):
        return False, 'Method schema file "%s" not found' % method_schema_path

    for filename, path, is_design_pattern in chain(
            map(lambda fn: (fn, os.path.join(schema_base_dir, fn), False), os.listdir(schema_base_dir)),
            map(lambda fn: (fn, os.path.join(design_pattern_schema_base_dir, fn), True),
                os.listdir(design_pattern_schema_base_dir))):
        if not filename.lower().endswith(".json"):
            continue
        schema_str = None
        schema = None
        try:
            with open(path, 'r') as f:
                schema_str = f.read()
        except (IOError, FileNotFoundError):
            return False, 'Error opening or reading schema file "%s"' % path
        try:
            schema = json.loads(schema_str)
        except json.JSONDecodeError:
            return False, 'Error parsing schema file "%s"' % path

        schema_store['design_patterns/' + filename if is_design_pattern else filename] = schema
        if filename.lower() == 'method.json':
            method_schema = schema

    return True, ''


def validate_method_schema(instance):
    global schema_resolver, schema_validator
    if method_schema is None:
        res, err_msg = init_schemata()
        if not res:
            return res, err_msg
    if schema_validator is None:
        schema_resolver = RefResolver.from_schema(method_schema, store=schema_store)
        schema_validator = Draft7Validator(method_schema, resolver=schema_resolver)
    try:
        instance['$schema'] = 'method.json'
        schema_validator.validate(instance)
    except ValidationError as e:
        return False, e.message
    return True, ''
