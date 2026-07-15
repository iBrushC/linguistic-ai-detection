import sys, os
sys.path.insert(0, '.')
os.environ['MPLBACKEND'] = 'Agg'
from itertools import combinations
from test import load_essays, group_by_author, author_name
from analysis import global_similarity, get_all_metrics

essays = load_essays()
grouped = group_by_author(essays)

print('=== Eisen pair detail (BM effect-size) ===')
for i, j in combinations(grouped['Erica X Eisen'], 2):
    a = get_all_metrics(essays[i]['body'])
    b = get_all_metrics(essays[j]['body'])
    r = global_similarity(a, b, method='brunnermunzel')
    t_i = essays[i]['title'].strip()[:35]
    t_j = essays[j]['title'].strip()[:35]
    print('  essays %d vs %d: sim=%.4f' % (i, j, r['similarity']))
    print('    titles: "%s" vs "%s"' % (t_i, t_j))
    for k, v in sorted(r['family_means'].items()):
        print('      family %-12s = %.4f' % (k, v))

print()
print('=== Green pair detail ===')
for i, j in combinations(grouped['Matthew Green'], 2):
    a = get_all_metrics(essays[i]['body'])
    b = get_all_metrics(essays[j]['body'])
    r = global_similarity(a, b, method='brunnermunzel')
    t_i = essays[i]['title'].strip()[:35]
    t_j = essays[j]['title'].strip()[:35]
    print('  essays %d vs %d: sim=%.4f' % (i, j, r['similarity']))
    print('    titles: "%s" vs "%s"' % (t_i, t_j))
    for k, v in sorted(r['family_means'].items()):
        print('      family %-12s = %.4f' % (k, v))
