# Import a data as CSV, split for rows
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

# The calc_counts function calculates any combination with births 
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
    
# min_max function finds the min and max of the births from the dataset
def min_max(data):
    return min(data, key = lambda x: x[4]), max(data, key = lambda x: x[4])

min_max(cdc_list)

# Compare if the births are increasing or decreasing and print the result e.g. "Births increased on 2001 from year 2000"
def check_birth_growth(birth_data_file):
    cdc_list = read_csv(birth_data_file)
    cdc_year_births = calc_counts(cdc_list, 0)
    previous_year_birth = 0
    previous_birth_diff = 0
    for year, total_births in cdc_year_births.items():
        current_year_birth = int(total_births)
        if previous_year_birth == 0:
            growth_status = "Growth of births in {} not available.".format(year)
            print(growth_status)
            previous_year_birth = current_year_birth
        else:
            if current_year_birth > previous_year_birth:
                growth_status = "Births increased in {}.".format(year)
                print(growth_status)
                previous_year_birth = current_year_birth
            elif current_year_birth < previous_year_birth:
                growth_status = "Births decreased in {}.".format(year)
                print(growth_status)
                previous_year_birth = current_year_birth
            elif current_year_birth == previous_year_birth:
                growth_status = "Births in {} was same as previous year.".format(year)
                print(growth_status)
                previous_year_birth = current_year_birth
    return previous_year_birth

check_birth_growth("US_births_1994-2003_CDC_NCHS.csv")
