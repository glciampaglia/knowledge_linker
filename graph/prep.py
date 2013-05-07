#!/usr/bin/env python

import re
from argparse import ArgumentParser
from contextlib import closing

if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('namespaces', help='tab-separated list of namespace codes')
    parser.add_argument('instances', help='list of instances')

    args = parser.parse_args()

    # read namespaces
    namespaces = {}
    with closing(open(args.namespaces)) as nsfile:
        for line in nsfile:
            ns, code = line.strip().split('\t')
            namespaces[ns] = code

    # build regexp
    x = re.compile('({})'.format('|'.join(namespaces.keys())))

    with closing(open(args.instances)) as ntfile:
        for line_no, line in enumerate(ntfile):
            if line.startswith('#'):
                continue
            # remove trailing newline and dot
            line = line.strip().strip('.').strip()
            # items is (entity predicate entity)
            items = line.split(' ')
            abbrv_items = []
            for item in items:
                item = item[1:-1]
                m = x.match(item)
                if m is not None:
                    matched_ns = m.group()
                    abbrv_ns = namespaces[matched_ns]
                    item = x.sub(abbrv_ns + ':', item)
                else:
                    raise ValueError('error: {}:{}: namespace not known: {}'.format(
                        args.instances, line_no + 1, item))
                abbrv_items.append(item)
                
            print ' '.join(abbrv_items)

            


