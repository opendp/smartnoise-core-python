import json

from jsonschema import validate


def validate_report(instance):

    with open('schema.json', 'r') as infile:
        schema = json.load(infile)

    return validate(instance=instance, schema=schema)


if __name__ == '__main__':
    report = {
        'accuracy': None,
        'algorithmInfo': {'argument': {'constraint': {'lowerbound': 0.0,
                                                      'upperbound': 100.0},
                                       'implementation': 'resize',
                                       'n': 1000},
                          'cite': '',
                          'mechanism': 'Automatic',
                          'name': ''},
        'description': 'DP release information',
        'nodeID': 11,
        'postprocess': False,
        'privacyLoss': {'delta': 0.0, 'epsilon': 0.65, 'name': 'approximate'},
        'releaseInfo': 44.808308258971856,
        'statistic': 'DPMean',
        'submission': 0,
        'variables': 'age'
    }
    print(validate_report({'self': 1}))
