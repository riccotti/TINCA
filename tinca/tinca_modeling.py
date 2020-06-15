import gzip
import glob
import json
import datetime
import numpy as np
import pandas as pd

from collections import defaultdict

from tinca.category_map import cod_mkt_cat2name

path = './'
path_data = './'
cat_level = 'categoria'

category_level = {
    'settore': 2,
    'reparto': 4,
    'categoria': 7,
    'sottocategoria': 9,
    'segmento': 11,
}


def get_all_years_months():
    months = list(range(1, 12 + 1))
    years = range(2007, 2016)
    all_years_months = sorted([(y, m) for m in months for y in years])
    return all_years_months


def get_demo_map(filename):
    dfdemo = pd.read_csv(filename)
    dfdemo.set_index('id_cliente', inplace=True)
    demo_map = dfdemo.to_dict('index')
    return demo_map


def get_item2category(filename, category_level=7):
    df = pd.read_csv(filename, delimiter=';', skipinitialspace=True)
    item2category = dict()
    for row in df.values:
        cod_mkt = str(row[1])
        cod_cat = cod_mkt[:category_level]
        product = cod_mkt_cat2name.get(cod_cat, None)
        item2category[str(row[0])] = product
    return item2category


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def get_average_frequency(shopping_date_list):
    if len(shopping_date_list) > 1:
        interleaving_time_list = list()
        for i in range(len(shopping_date_list) - 1):
            d1 = shopping_date_list[i]
            d2 = shopping_date_list[i + 1]
            interleaving_time_list.append((d2 - d1).days)
        avg_frequency = np.mean(interleaving_time_list)
    else:
        avg_frequency = 0
    return avg_frequency


def main():

    customer_type = 'italian'
    max_val = 1000.0

    if customer_type == 'italian':
        print(datetime.datetime.now(), 'Reading italian demographics')
        demo_map = get_demo_map(path + 'italians_demographics.csv')
        template_filaname = 'ita_new_batch_*.json.gz'
        print_nbr = 10000
    else:
        print(datetime.datetime.now(), 'Reading foreign demographics')
        demo_map = get_demo_map(path + 'foreign_demographics.csv')
        template_filaname = 'for_batch_*.json.gz'
        print_nbr = 100

    print(datetime.datetime.now(), 'Reading marketing')
    item2category = get_item2category(path + 'market.csv', category_level[cat_level])

    for filename in glob.glob(path_data + template_filaname):

        print(datetime.datetime.now(), filename)
        filedata = gzip.open(filename, 'r')
        count_row = 0
        for row in filedata:
            jdata = json.loads(row)

            customer_id = jdata['customer_id']
            country = demo_map[customer_id]['stato']
            membership_date = demo_map[customer_id]['anno_socio']
            data = jdata['data']
            if customer_id not in demo_map:
                continue

            year_month_basketid = defaultdict(list)

            # analyze the data of a single customer
            for basketid, record in data.items():
                anno = int(record['year'])
                mese = int(record['month'])
                anno_mese = str((anno, mese))
                year_month_basketid[anno_mese].append(basketid)

            year_month_basketid = default_to_regular(year_month_basketid)

            customer_model = dict()
            customer_model['customer_id'] = customer_id
            customer_model['country'] = country
            customer_model['membership_date'] = membership_date

            for year_month in sorted(year_month_basketid):
                customer_ym_model = dict()
                # print(year_month, end=' ')
                basket_ids = sorted(year_month_basketid[year_month])

                # print(len(basket_ids))

                nbr_baskets = 0
                nbr_products = 0
                distinct_products = set()
                total_expenditure = 0.0
                basket_len_list = list()
                shopping_date_list = list()
                product_nbr_baskets = dict()
                product_expenditure = dict()
                product_shopping_date_list = defaultdict(list)
                for bid in basket_ids:
                    # print(bid)
                    basket = data[bid]
                    # print(basket)

                    basket_products = dict()
                    for id_mkt in basket['basket']:
                        if id_mkt == '##' or id_mkt == 'null':
                            continue
                        product = item2category.get(id_mkt, None)
                        # print(id_mkt, product)
                        if product == '##' or product is None:
                            continue
                        importo = basket['basket'][id_mkt][0]
                        qta = basket['basket'][id_mkt][1]
                        exp = max(importo * qta, 0.0)
                        if product not in basket_products:
                            basket_products[product] = 0.0
                        basket_products[product] += exp

                    if len(basket_products) == 0:
                        continue

                    date = datetime.datetime.strptime(basket['date'], '%Y-%m-%d %H:%M:%S')
                    shopping_date_list.append(date)

                    nbr_baskets += 1
                    basket_len_list.append(len(basket_products))
                    for product, exp in basket_products.items():
                        nbr_products += 1
                        distinct_products.add(product)
                        exp = min(exp, max_val)
                        total_expenditure += exp
                        if product not in product_expenditure:
                            product_nbr_baskets[product] = 0
                            product_expenditure[product] = 0.0
                        product_nbr_baskets[product] += 1
                        product_expenditure[product] += exp
                        product_shopping_date_list[product].append(date)

                if nbr_baskets == 0:
                    break

                nbr_distinct_products = len(distinct_products)
                avg_basket_len = np.mean(basket_len_list)
                avg_expenditure = total_expenditure / nbr_baskets
                avg_frequency = get_average_frequency(shopping_date_list)
                product_avg_frequency = dict()
                for product, sdl in product_shopping_date_list.items():
                    product_avg_frequency[product] = get_average_frequency(sdl)

                # print('nbr_baskets', nbr_baskets)
                # print('nbr_products', nbr_products)
                # print('nbr_distinct_products', nbr_distinct_products)
                # print('total_expenditure', total_expenditure)
                # print('avg_basket_len', avg_basket_len)
                # print('avg_expenditure', avg_expenditure)
                # print('avg_frequency', avg_frequency)
                # for product in product_nbr_baskets:
                #     print(product, product_nbr_baskets[product], product_expenditure[product], product_avg_frequency[product])

                # print('')
                customer_ym_model['nbr_baskets'] = nbr_baskets
                customer_ym_model['nbr_products'] = nbr_products
                customer_ym_model['nbr_distinct_products'] = nbr_distinct_products
                customer_ym_model['total_expenditure'] = float(np.round(total_expenditure, 2))
                customer_ym_model['avg_basket_len'] = float(np.round(avg_basket_len, 2))
                customer_ym_model['avg_expenditure'] = float(np.round(avg_expenditure, 2))
                customer_ym_model['avg_frequency'] = float(np.round(avg_frequency, 2))
                for product in product_nbr_baskets:
                    customer_ym_model['%s_nbr_baskets' % product] = product_nbr_baskets[product]
                    customer_ym_model['%s_expenditure' % product] = float(np.round(product_expenditure[product], 2))
                    customer_ym_model['%s_avg_frequency' % product] = float(np.round(product_avg_frequency[product], 2))

                customer_model[year_month] = customer_ym_model

            # print(json.dumps(customer_model))

            json_str = '%s\n' % json.dumps(customer_model)
            json_bytes = json_str.encode('utf-8')
            with gzip.GzipFile(path + '%s_model.json.gz' % customer_type, 'a') as fout:
                fout.write(json_bytes)

            if count_row % print_nbr == 0:
                print(datetime.datetime.now(), count_row)

            count_row += 1
        #     break
        # break


if __name__ == "__main__":
    main()
