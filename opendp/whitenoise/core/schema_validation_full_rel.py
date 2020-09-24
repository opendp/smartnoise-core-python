import json
from jsonschema import validate

schema_f = open("release-schema.json")
schema = json.load(schema_f)
instance = [{'description': 'DP release information', 'variables': 'income', 'statistic': 'DPMaximum', 'releaseInfo': 2743921.2338242335, 'privacyLoss': {'delta': 0.0, 'epsilon': 0.65, 'name': 'approximate'}, 'accuracy': None, 'submission': 0, 'nodeID': 21, 'postprocess': False, 'algorithmInfo': {'mechanism': 'Automatic', 'name': '', 'cite': '', 'argument': {'constraint': {'lowerbound': 0.0, 'upperbound': 1000000.0}}}}, {'description': 'DP release information', 'variables': 'age', 'statistic': 'DPMean', 'releaseInfo': 45.52654758068556, 'privacyLoss': {'delta': 0.0, 'epsilon': 0.65, 'name': 'approximate'}, 'accuracy': None, 'submission': 0, 'nodeID': 11, 'postprocess': False, 'algorithmInfo': {'mechanism': 'Automatic', 'name': '', 'cite': '', 'argument': {'constraint': {'lowerbound': 0.0, 'upperbound': 100.0}, 'implementation': 'resize', 'n': 1000}}}]
validate(instance, schema)
