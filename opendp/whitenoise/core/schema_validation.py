import json

from jsonschema import validate


def validate_report(instance):

    with open('schema.json', 'r') as infile:
        schema = json.load(infile)

    return validate(instance=instance, schema=schema)


if __name__ == '__main__':

    # This does not pass validation
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

    # This does pass validation
    passing_report = {
        'self': {
            'schema_version': "0",
            'created': "2020-1-1",
        },
        'dataset': {
            'unitOfAnalysis': 'unit',
            'structure': 'wide',
            'rowCount': 0,
            'variableCount': 1
        },
        'dp-releases': [
            {
                'statistic': 'mean',
                'submission': 0,
                'variables': {
                    'arbitrary-key': 'age'
                },
                'releaseInfo': {
                    'mechanism': 'Laplace',
                },
                'nodeID': 11,
                'postprocess': False,
                'accuracy': {
                    'accuracyValue': 100.0
                },
                'batch': 0,
                'privacyLoss': {'delta': 0.0, 'epsilon': 0.65, 'name': 'approx dp'},
                'algorithm': {'lowerbound': 0.0,
                              'upperbound': 100.0,
                              'implementation': 'resize',
                              'n': 1000,
                              'cite': '',
                              'name': ''},
            }

        ],
    }

    validate_report(passing_report)
    print("Validation succeeded")
