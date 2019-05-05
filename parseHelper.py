import ast

def parseJson(json_string):
    json_string = json_string.replace('NA','""')
    json_string = json_string.replace('TRUE',"True")
    json_string = json_string.replace('FALSE',"False")
    dic = ast.literal_eval(json_string)

    features = ['status_last_archived_0_24m', 'num_arch_ok_0_12m', 'status_3rd_last_archived_0_24m', 'account_worst_status_0_3m', 'num_unpaid_bills', 'status_max_archived_0_24_months', 'num_arch_ok_12_24m', 'age', 'num_active_div_by_paid_inv_0_12m', 'avg_payment_span_0_12m', 'max_paid_inv_0_24m', 'status_2nd_last_archived_0_24m', 'account_status', 'merchant_group', 'sum_paid_inv_0_12m', 'max_paid_inv_0_12m', 'time_hours']

    headers = ""
    data = ""

    for feature in features:
        headers+='"'+feature+'";'
        try:
            data+=str(dic[feature])+";"
        except KeyError:
            data+="NA;"


    headers = headers[:-1]+"\n"
    data = data[:-1]


    return headers+data