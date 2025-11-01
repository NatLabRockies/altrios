import polars as pl

def calculate_oc_number() -> int:
    return 40

def calculate_truck_number(df):
    return (df
        .with_columns(
            pl.max_horizontal(
                pl.col("full_cars"),
                calculate_oc_number()
            ).alias("truck_number"))
    )

def truck_resource(df):
    total_truck_number = df.select(pl.col("truck_number").sum()).item()
    return total_truck_number

#terminal = 'Allouez'
#train_data_as_dict = schedule.build_train_timetable(terminal, swap_arrive_depart = True, as_dicts = True)
#print(train_data_as_dict)