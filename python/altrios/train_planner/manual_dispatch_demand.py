import polars as pl
from altrios.utilities import package_root


class manual_dispatch_demand:
    def __init__(self, filepath, car_type_pwr={}, sim_set={}):
        self.dispatch_filepath = filepath
        if not car_type_pwr:
            self.car_type_pwr = [
                {"Car_Type": "Freight", "details": {
                    "Tons_Per_Car_Loaded": 132.2772,
                    "Tons_Per_Car_Empty": 32.518145,
                    "HP_Required_Per_Ton_Loaded": 2,
                    "HP_Required_Per_Ton_Empty": 2}},
                {"Car_Type": "Intermodal", "details": {
                    "Tons_Per_Car_Loaded": 49.3, # 146.60723,
                    "Tons_Per_Car_Empty": 31.0, # 34.722765,
                    "HP_Required_Per_Ton_Loaded": 3, #4,
                    "HP_Required_Per_Ton_Empty": 3}},#4}},
                {"Car_Type": "Manifest", "details": {
                    "Tons_Per_Car_Loaded": 146.60723,
                    "Tons_Per_Car_Empty": 34.722765,
                    "HP_Required_Per_Ton_Loaded": 1.5,
                    "HP_Required_Per_Ton_Empty": 1.5}},
                {"Car_Type": "Unit", "details": {
                    "Tons_Per_Car_Loaded": 132.2772,
                    "Tons_Per_Car_Empty": 32.518145,
                    "HP_Required_Per_Ton_Loaded": 2,
                    "HP_Required_Per_Ton_Empty": 2}}
            ]
        else:
            self.car_type_pwr = car_type_pwr

        if not sim_set:
            self.sim_set = {
                "Cars_Per_Train_Min": 60,
                "Cars_Per_Train_Target": 90,
                "Number_of_Days": 21
            }
        else:
            self.sim_set = sim_set

    def demand_from_dispatch(self):

        dispatch_schedule = pl.read_csv(self.dispatch_filepath)

        node_list = dispatch_schedule["Origin"].unique()
        demand = dispatch_schedule \
            .group_by(["Origin", "Destination", "Train_Type"]) \
            .agg([pl.col("Cars_Loaded").sum().alias("Number_of_Cars_Loaded"),
                  pl.col("Cars_Empty").sum().alias("Number_of_Cars_Empty"),
                  (pl.col("Train_Type").len()).alias("Number_of_Trains")])
        demand = demand.with_columns(
            (pl.col("Number_of_Cars_Loaded") + pl.col("Number_of_Cars_Empty")).alias("Number_of_Cars"),
            (pl.col("Number_of_Cars_Loaded") * 2).alias("Number_of_Containers_Loaded"),
            (pl.col("Number_of_Cars_Empty") * 2).alias("Number_of_Containers_Empty"))
        demand = demand.join(
            pl.DataFrame(self.car_type_pwr).unnest("details"),
            left_on="Train_Type", right_on="Car_Type", how="left")
        demand = demand.join(
            pl.DataFrame([{"Origin": ori, "details": self.sim_set} for ori in node_list.to_list()]) \
                .unnest("details"), left_on="Origin", right_on="Origin", how="left")

        return dispatch_schedule, demand, node_list


if __name__ == "__main__":
    dispatch_schedule_path = (package_root()).parents[1] / "IntermediateFiles/dispatch_schedule.txt"
    dispatch_scheduler = manual_dispatch_demand(dispatch_schedule_path)
    dispatch_schedule, demand, node_list = dispatch_scheduler.demand_from_dispatch()