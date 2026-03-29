import pandas as pd
import math
#from scipy import stats
#import openpyxl
#from scipy.sparse import data

id_list = [123656, 123009,
    261306914,261304035,261307332,261307333,261307335,261307336,260236309,261307337,261307338,
    261306918,261356005,261273807,260489695,261305436,261307603,261303049,261307493,261307341,
    260581198,261307812,261307342,261305540,261307489,261266323,261302036,261306911,260694322,
    261250028,261307344,261132166,261278480,261307345,261248222,261301996,261266334]

def get_row_col(id_value):
    "Get the row/columns starting point for the assignment based on McGill ID"
    # Convert to string to extract digits
    id_str = str(id_value)

    # Row number: last 2 digits / 2, rounded up
    last_two_digits = int(id_str[-2:])
    row_number = math.ceil(last_two_digits / 2)

    # Column number: 3rd digit from last + 1
    third_from_last = int(id_str[-3])
    column_number = third_from_last + 1

    return row_number, column_number


def main():

    target_part = "A"

    for input_id in id_list:
        target_row, target_column = get_row_col(input_id)

        print(f"McGill ID: {input_id} Selection is [{target_row} {target_column} {target_part}]")

        xls = pd.ExcelFile('Lab 1 Data.xlsx')

        # xlsx_rand_num = pd.read_excel(xls, 'Random Number Table', header=0, names=['Row', 'Column', 'Rand', 'A', 'B', 'C'])
        xlsx_rand_num = pd.concat([pd.read_excel(xls, 'Random Number Table', header=0,
                        names=['Row', 'Column', 'Rand', 'A', 'B', 'C'])] * 2, ignore_index=True)

        xlsx_population = pd.read_excel(xls, 'Population', header=None, names=['ID', 'Gender', 'Score'])

        indexed_data = pd.merge(left=xlsx_rand_num, right=xlsx_population, left_on=target_part, right_on="ID")

        data = indexed_data.loc[((indexed_data['Row'] == target_row) & (indexed_data['Column'] == target_column)).idxmax():] \
            .drop_duplicates(subset=[target_part]).head(60)

        exportable_data = data.loc[:,["Gender", "Score"]]
        # exportable_data.to_csv(f'{input_id}_Row{target_row}_Col{target_column}_{target_part}.csv', index=False)
        exportable_data.to_excel(f'{input_id}_Row{target_row}_Col{target_column}_{target_part}.xlsx', index=False)

        c,m, s = data['Score'].describe()[0:3]
        print("Count:", c, "Mean", round(m,2), "SD", round(s,2))
        print()

if __name__ == '__main__':
    main()
