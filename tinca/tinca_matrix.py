import gzip
import glob
import json
import datetime
import numpy as np
import pandas as pd

from collections import defaultdict

from tinca.category_map import cod_mkt_cat2name

path = './'


def model2vector(ym_model, native, products):
    vector = dict()
    vector['is_native'] = native
    vector['nbr_baskets'] = ym_model['nbr_baskets']
    vector['nbr_products'] = ym_model['nbr_products']
    vector['nbr_distinct_products'] = ym_model['nbr_distinct_products']
    vector['total_expenditure'] = ym_model['total_expenditure']
    vector['avg_basket_len'] = ym_model['avg_basket_len']
    vector['avg_expenditure'] = ym_model['avg_expenditure']
    vector['avg_frequency'] = ym_model['avg_frequency']
    for product in products:
        if '%s_nbr_baskets' % product in ym_model:
            vector['%s_nbr_baskets' % product] = ym_model['%s_nbr_baskets' % product]
            vector['%s_expenditure' % product] = ym_model['%s_expenditure' % product]
            vector['%s_avg_frequency' % product] = ym_model['%s_avg_frequency' % product]
        else:
            vector['%s_nbr_baskets' % product] = 0
            vector['%s_expenditure' % product] = 0.0
            vector['%s_avg_frequency' % product] = 0.0

    return vector


# nbr shop per year foreign
# 2011 139076
# 2012 139868
# 2013 143711
# 2014 145092
# 2015 143000
# 2016 82658
# 2010 135206
# 2009 131101
# 2008 119574

def main():

    products = sorted(cod_mkt_cat2name.values())

    native = 0
    print_nbr = 100
    count_row = 0
    header = set()
    filedata = gzip.open(path + 'foreign_model.json.gz', 'r')
    for row in filedata:
         customer_model = json.loads(row)
         for key in customer_model:
             if key in ['customer_id', 'country', 'membership_date']:
                 continue
             year_month = key
             ym_model = customer_model[year_month]
             vector = model2vector(ym_model, native, products)
             vector['customer_id'] = customer_model['customer_id']
             vector['country'] = customer_model['country']
             vector['membership_date'] = customer_model['membership_date']
             df = pd.DataFrame(data=[vector])
             header_flag = year_month not in header
             vector_str = df.to_csv(header=header_flag, index=False)
             vector_bytes = vector_str.encode('utf-8')
             year_month_str = year_month.replace('(', '').replace(')', '').replace(',', '_')
             with gzip.GzipFile(path + 'foreign_matrix_%s.csv.gz' % year_month_str, 'a') as fout:
                 fout.write(vector_bytes)
             header.add(year_month)
    
         if count_row % print_nbr == 0:
             print(datetime.datetime.now(), count_row)
    
         count_row += 1

    batch_id = 7
    gap = 100000
    start_from = gap * batch_id
    stop_at = start_from + gap
    started = False

    native = 1
    print_nbr = 10000
    count_row = 0
    header = set()
    filedata = gzip.open(path + 'italian_model.json.gz', 'r')
    for row in filedata:

        if count_row % print_nbr == 0:
            print(datetime.datetime.now(), count_row, started)

        if count_row <= start_from:
            count_row += 1
            continue

        if count_row > stop_at:
            break

        started = True
        customer_model = json.loads(row)
        for key in customer_model:
            if key in ['customer_id', 'country', 'membership_date']:
                continue
            year_month = key
            ym_model = customer_model[year_month]
            vector = model2vector(ym_model, native, products)
            vector['customer_id'] = customer_model['customer_id']
            vector['country'] = customer_model['country']
            vector['membership_date'] = customer_model['membership_date']
            df = pd.DataFrame(data=[vector])
            header_flag = False  # year_month not in header
            vector_str = df.to_csv(header=header_flag, index=False)
            vector_bytes = vector_str.encode('utf-8')
            year_month_str = year_month.replace('(', '').replace(')', '').replace(',', '_').replace(' ', '')
            with gzip.GzipFile(path + 'italian_matrix_%s_batch%s.csv.gz' % (year_month_str, batch_id), 'a') as fout:
                fout.write(vector_bytes)
            header.add(year_month)

        count_row += 1

if __name__ == "__main__":
    main()
