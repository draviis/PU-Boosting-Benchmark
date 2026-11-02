def comparator_ids(cell):
    assert cell['scheme'] in ('one_sample','two_sample')
    prefix='TM' if cell['scheme']=='two_sample' else 'M'
    return [(prefix+'020','RN_CatBoost'),(prefix+'025','CostSensitive_CatBoost')]
