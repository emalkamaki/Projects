csv_list = open("US_births_1994-2003_CDC_NCHS.csv").read().split("\n")

def read_csv(filename):
    string_data = open(filename).read()
    string_list = string_data.split("\n")[1:]
    final_list = []
    
    for row in string_list:
        string_fields = row.split(",")
        int_fields = []
        for value in string_fields:
            int_fields.append(int(value))
        final_list.append(int_fields)
    return final_list
        
cdc_list = read_csv("US_births_1994-2003_CDC_NCHS.csv")

def calc_counts(data, column):
    births_per_data = {}
    for row in data:
        x = row[column]
        births = row[4]
        if x in births_per_data:
            births_per_data[x] = births_per_data[x] + births
        else:
            births_per_data[x] = births
    return births_per_data

cdc_year_births = calc_counts(cdc_list, 0)
cdc_year_births

cdc_month_births = calc_counts(cdc_list, 1)
cdc_month_births

cdc_dom_births = calc_counts(cdc_list, 2)
cdc_dom_births

cdc_dow_births = calc_counts (cdc_list, 3)
cdc_dow_births
    
