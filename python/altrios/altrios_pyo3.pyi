"""Type stubs for ALTRIOS Python bindings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

class SerdeAPI:
    """Base class representing serializable objects in ALTRIOS."""

    ACCEPTED_BYTE_FORMATS: ClassVar[list[str]]
    ACCEPTED_STR_FORMATS: ClassVar[list[str]]

    # Class methods for deserialization
    @classmethod
    def from_bincode(cls, bincode_data: bytes, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_str(cls, contents: str, fmt: str, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_file(cls, filepath: str | Path, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_json(cls, json_str: str, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_yaml(cls, yaml_str: str, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_toml(cls, toml_str: str, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_resource(cls, filepath: str | Path, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_msg_pack(cls, msg_pack: bytes, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_pydict(
        cls,
        pydict: dict[str, Any],
        data_fmt: str = "msg_pack",
        skip_init: bool = False,
    ) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...

    # Instance methods for serialization
    def to_file(self, filepath: str | Path) -> None: ...
    def to_str(self, fmt: str) -> str: ...
    def to_bincode(self) -> bytes: ...
    def to_json(self) -> str: ...
    def to_yaml(self) -> str: ...
    def to_toml(self) -> str: ...
    def to_msg_pack(self) -> bytes: ...
    def to_pydict(self, data_fmt: str = "msg_pack", flatten: bool = False) -> dict[str, Any]: ...

    # Cloning/copy methods
    def __copy__(self) -> Self: ...
    def copy(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def clone(self) -> Self: ...

@dataclass
class Consist(SerdeAPI):
    assert_limits: bool
    history: ConsistStateHistoryVec
    loco_vec: list[Locomotive]
    save_interval: int
    state: ConsistState

    def __init__(
        self,
        loco_vec: list[Locomotive],
        save_interval: int | None = None,
    ) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def get_save_interval(self) -> int: ...
    def set_pdct_prop(self) -> None: ...
    def set_pdct_resgreedy(self) -> None: ...
    def set_save_interval(self, save_interval: int): ...
    def __copy__(self) -> Self: ...

class ConsistSimulation(SerdeAPI):
    i: int
    loco_con: Consist
    power_trace: PowerTrace
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def get_save_interval(self) -> int: ...
    def set_save_interval(self, save_interval: int): ...
    def walk(self) -> None: ...
    def __copy__(self) -> Self: ...

class ConsistState(SerdeAPI):
    energy_fuel_joules: float
    energy_out_joules: float
    energy_res_joules: float
    i: int
    pwr_dyn_brake_max_watts: float
    pwr_fuel_watts: float
    pwr_out_max_non_reves_watts: float
    pwr_out_max_reves_watts: float
    pwr_out_max_watts: float
    pwr_out_req_watts: float
    pwr_out_unfulfilled_watts: float
    pwr_out_watts: float
    pwr_rate_out_max_watts_per_second: float
    pwr_regen_max_watts: float
    pwr_regen_unfulfilled_watts: float
    pwr_reves_watts: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class ConsistStateHistoryVec(SerdeAPI):
    energy_fuel_joules: list[float]
    energy_out_joules: list[float]
    energy_res_joules: list[float]
    i: list[int]
    pwr_dyn_brake_max_watts: list[float]
    pwr_fuel_watts: list[float]
    pwr_out_max_non_reves_watts: list[float]
    pwr_out_max_reves_watts: list[float]
    pwr_out_max_watts: list[float]
    pwr_out_req_watts: list[float]
    pwr_out_unfulfilled_watts: list[float]
    pwr_out_watts: list[float]
    pwr_rate_out_max_watts_per_second: list[float]
    pwr_regen_max_watts: list[float]
    pwr_regen_unfulfilled_watts: list[float]
    pwr_reves_watts: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

class ElectricDrivetrain(SerdeAPI):
    eta_interp: list[float]
    history: ElectricDrivetrainStateHistoryVec
    pwr_in_frac_interp: list[float]
    pwr_out_frac_interp: list[float]
    pwr_out_max_watts: float
    save_interval: int | None
    state: ElectricDrivetrainState

    def __init__(
        self,
        pwr_out_frac_interp: list[float],
        eta_interp: list[float],
        pwr_out_max_watts: float,
        save_interval: int | None = None,
    ) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class ElectricDrivetrainState(SerdeAPI):
    energy_elec_dyn_brake_joules: float
    energy_elec_prop_in_joules: float
    energy_loss_joules: float
    energy_mech_dyn_brake_joules: float
    energy_mech_prop_out_joules: float
    eta: float
    i: int
    pwr_elec_dyn_brake_watts: float
    pwr_elec_prop_in_watts: float
    pwr_loss_watts: float
    pwr_mech_dyn_brake_watts: float
    pwr_mech_out_max_watts: float
    pwr_mech_prop_out_watts: float
    pwr_mech_regen_max_watts: float
    pwr_rate_out_max_watts_per_second: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class ElectricDrivetrainStateHistoryVec(SerdeAPI):
    energy_elec_dyn_brake_joules: list[float]
    energy_elec_prop_in_joules: list[float]
    energy_loss_joules: list[float]
    energy_mech_dyn_brake_joules: list[float]
    energy_mech_prop_out_joules: list[float]
    eta: list[float]
    i: list[int]
    pwr_elec_dyn_brake_watts: list[float]
    pwr_elec_prop_in_watts: list[float]
    pwr_loss_watts: list[float]
    pwr_mech_dyn_brake_watts: list[float]
    pwr_mech_out_max_watts: list[float]
    pwr_mech_prop_out_watts: list[float]
    pwr_mech_regen_max_watts: list[float]
    pwr_rate_out_max_watts_per_second: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

class FuelConverter(SerdeAPI):
    eta_interp: list[float]
    history: FuelConverterStateHistoryVec
    pwr_out_frac_interp: list[float]
    pwr_out_max_watts: float
    pwr_idle_fuel_watts: float
    pwr_ramp_lag_seconds: float
    save_interval: int | None
    state: FuelConverterState

    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def set_default_elev_and_temp_derate(self) -> None: ...
    def __copy__(self) -> Self: ...

class FuelConverterState(SerdeAPI):
    energy_brake_joules: float
    energy_fuel_joules: float
    energy_idle_fuel_joules: float
    energy_loss_joules: float
    engine_on: bool
    eta: float
    i: int
    pwr_fuel_watts: float
    pwr_brake_watts: float
    pwr_idle_fuel_watts: float
    pwr_loss_watts: float
    pwr_out_max_watts: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class FuelConverterStateHistoryVec(SerdeAPI):
    energy_brake_joules: list[float]
    energy_fuel_joules: list[float]
    energy_idle_fuel_joules: list[float]
    energy_loss_joules: list[float]
    engine_on: list[bool]
    eta: list[float]
    i: list[int]
    pwr_brake_watts: list[float]
    pwr_fuel_watts: list[float]
    pwr_idle_fuel_watts: list[float]
    pwr_loss_watts: list[float]
    pwr_out_max_watts: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

class Generator(SerdeAPI):
    eta_interp: list[float]
    history: GeneratorStateHistoryVec
    pwr_in_frac_interp: list[float]
    pwr_out_frac_interp: list[float]
    pwr_out_max_watts: float
    save_interval: int | None
    state: GeneratorState

    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class GeneratorState(SerdeAPI):
    energy_elec_aux_joules: float
    energy_elec_prop_out_joules: float
    energy_loss_joules: float
    energy_mech_in_joules: float
    eta: float
    i: int
    pwr_elec_aux_watts: float
    pwr_elec_out_max_watts: float
    pwr_elec_prop_out_watts: float
    pwr_loss_watts: float
    pwr_mech_in_watts: float
    pwr_rate_out_max_watts_per_second: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class GeneratorStateHistoryVec(SerdeAPI):
    energy_elec_aux_joules: list[float]
    energy_elec_prop_out_joules: list[float]
    energy_loss_joules: list[float]
    energy_mech_in_joules: list[float]
    eta: list[float]
    i: list[int]
    pwr_elec_aux_watts: list[float]
    pwr_elec_out_max_watts: list[float]
    pwr_elec_prop_out_watts: list[float]
    pwr_loss_watts: list[float]
    pwr_mech_in_watts: list[float]
    pwr_rate_out_max_watts_per_second: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

@dataclass
class LocoParams(SerdeAPI):
    mass_kilograms: float | None = 0.0
    brake_count: int = 0
    pwr_aux_offset_watts: float | None = None
    pwr_aux_traction_coeff_ratio: float | None = None
    force_max_newtons: float | None = None

    @classmethod
    def from_dict(cls, param_dict: dict[str, float]) -> Self: ...
    def to_dict(self) -> dict[str, float]: ...

@dataclass
class ConventionalLoco(SerdeAPI):
    pass

@dataclass
class HybridLoco(SerdeAPI):
    pass

@dataclass
class BatteryElectricLoco(SerdeAPI):
    pass

@dataclass
class DummyLoco(SerdeAPI):
    pass

class Locomotive(SerdeAPI):
    assert_limits: bool
    edrv: ElectricDrivetrain
    fc: FuelConverter
    fuel_res_ratio: float
    fuel_res_split: float
    gen: Generator
    history: LocomotiveStateHistoryVec
    pwr_aux_watts: float
    res: ReversibleEnergyStorage
    save_interval: int
    state: LocomotiveState

    @classmethod
    def __new__(
        cls,
        loco_type: ConventionalLoco | HybridLoco | BatteryElectricLoco | DummyLoco,
        loco_params: LocoParams,
    ): ...
    @classmethod
    def default_battery_electric_loco(cls) -> Locomotive: ...
    @classmethod
    def default_hybrid_electric_loco(cls) -> Locomotive: ...
    @classmethod
    def build_conventional_loco(
        cls,
        fuel_converter: FuelConverter,
        generator: Generator,
        drivetrain: ElectricDrivetrain,
        loco_params: LocoParams,
        save_interval: int | None,
    ) -> Self: ...
    @classmethod
    def build_dummy_loco(cls) -> Self: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def get_save_interval(self) -> Any: ...
    def set_save_interval(self, save_interval: int): ...
    def __copy__(self) -> Self: ...

class LocomotiveSimulation(SerdeAPI):
    i: int
    loco_unit: Locomotive
    power_trace: PowerTrace

    def __init__(
        self,
        loco_unit: Locomotive,
        power_trace: PowerTrace,
        save_interval: int | None = None,
    ) -> None: ...
    def clone(self) -> Self: ...
    def get_save_interval(self) -> int: ...
    def set_save_interval(self, save_interval: int): ...
    def walk(self) -> None: ...
    def __copy__(self) -> Self: ...

class LocomotiveState(SerdeAPI):
    energy_aux_joules: float
    energy_out_joules: float
    i: int
    pwr_aux_watts: float
    pwr_out_max_watts: float
    pwr_out_watts: float
    pwr_rate_out_max_watts_per_second: float
    pwr_regen_max_watts: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class LocomotiveStateHistoryVec(SerdeAPI):
    energy_aux_joules: list[float]
    energy_out_joules: list[float]
    i: list[int]
    pwr_aux_watts: list[float]
    pwr_out_max_watts: list[float]
    pwr_out_watts: list[float]
    pwr_rate_out_max_watts_per_second: list[float]
    pwr_regen_max_watts: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

@dataclass
class PowerTrace(SerdeAPI):
    time_seconds: list[float]
    pwr_watts: list[float]
    engine_on: list[bool] | None

    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...
    @classmethod
    def from_csv_file(cls, pathstr: str | Path) -> Self: ...
    def to_csv_file(self, pathstr: str | Path): ...

@dataclass
class SpeedTrace(SerdeAPI):
    time_seconds: list[float]
    speed_meters_per_second: list[float]
    engine_on: list[bool] | None

    def __init__(
        self,
        time_seconds: list[float],
        speed_meters_per_second: list[float],
        engine_on: list[bool] | None = None,
    ) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...
    @classmethod
    def from_csv_file(cls, pathstr: str | Path) -> Self: ...
    def to_csv_file(self, pathstr: str | Path): ...

class ReversibleEnergyStorage(SerdeAPI):
    energy_capacity_joules: float
    eta_interp_values: list[list[list[float]]]
    history: ReversibleEnergyStorageStateHistoryVec
    max_soc: float
    min_soc: float
    pwr_out_max_watts: float
    save_interval: int | None
    soc_hi_ramp_start: float | None
    soc_lo_ramp_start: float | None
    state: ReversibleEnergyStorageState

    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class ReversibleEnergyStorageState(SerdeAPI):
    energy_aux_joules: float
    energy_loss_joules: float
    energy_out_chemical_joules: float
    energy_out_electrical_joules: float
    energy_out_propulsion_joules: float
    eta: float
    i: int
    max_soc: float
    min_soc: float
    pwr_aux_watts: float
    pwr_loss_watts: float
    pwr_out_chemical_watts: float
    pwr_out_electrical_watts: float
    pwr_out_max_watts: float
    pwr_out_propulsion_watts: float
    pwr_regen_max_watts: float
    soc: float
    soc_hi_ramp_start: float
    soc_lo_ramp_start: float
    soh: float
    temperature_celsius: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class ReversibleEnergyStorageStateHistoryVec(SerdeAPI):
    energy_aux_joules: list[float]
    energy_loss_joules: list[float]
    energy_out_chemical_joules: list[float]
    energy_out_electrical_joules: list[float]
    energy_out_propulsion_joules: list[float]
    eta: list[float]
    i: list[int]
    max_soc: list[float]
    min_soc: list[float]
    pwr_aux_watts: list[float]
    pwr_loss_watts: list[float]
    pwr_out_chemical_watts: list[float]
    pwr_out_electrical_watts: list[float]
    pwr_out_max_watts: list[float]
    pwr_out_propulsion_watts: list[float]
    pwr_regen_max_watts: list[float]
    soc: list[float]
    soc_hi_ramp_start: list[float]
    soc_lo_ramp_start: list[float]
    soh: list[float]
    temperature_celsius: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

class SetSpeedTrainSim(SerdeAPI):
    loco_con: Consist
    state: TrainState
    speed_trace: SpeedTrace
    history: TrainStateHistoryVec
    i: int
    save_interval: int | None

    @classmethod
    def __init__(
        cls,
        loco_con: Consist,
        state: TrainState,
        train_res_file: str | None,
        path_tpc_file: str | None,
        speed_trace: SpeedTrace,
        save_interval: int | None,
        simulation_days: int | None,
    ) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def set_save_interval(self, save_interval: int): ...
    def walk(self) -> None: ...
    def to_dataframe(self) -> Any: ...

@dataclass
class TrainState:
    time_seconds: float
    i: int
    offset_meters: float
    offset_back_meters: float
    total_dist_meters: float
    link_idx_front: int
    offset_in_link_meters: float
    grade_front: float
    speed_meters_per_second: float
    speed_limit_meters_per_second: float
    speed_target_meters_per_second: float
    dt_seconds: float
    length_meters: float
    mass_static_kilograms: float
    mass_rot_kilograms: float
    mass_freight_kilograms: float
    weight_static_newtons: float
    res_rolling_newtons: float
    res_bearing_newtons: float
    res_davis_b_newtons: float
    res_aero_newtons: float
    res_grade_newtons: float
    res_curve_newtons: float
    elev_front_meters: float
    pwr_res_watts: float
    pwr_accel_watts: float
    pwr_whl_out_watts: float
    energy_whl_out_joules: float
    energy_whl_out_pos_joules: float
    energy_whl_out_neg_joules: float
    @classmethod
    def default(cls) -> TrainState: ...
    @classmethod
    def from_json(cls, json_str: str) -> TrainState: ...
    @classmethod
    def __new__(
        cls,
        length_meters: float,
        mass_static_kilograms: float,
        mass_adj_kilograms: float,
        mass_freight_kilograms: float,
        init_train_state: InitTrainState | None,
    ) -> Self: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_yaml(cls, yaml_str: str) -> TrainState: ...
    def to_yaml(self) -> str: ...
    def clone(self) -> TrainState: ...
    def reset_orphaned(self) -> None: ...

class TrainStateHistoryVec(SerdeAPI):
    time_seconds: list[float]
    offset_meters: list[float]
    offset_back_meters: list[float]
    link_idx_front: list[int]
    offset_in_link_meters: list[float]
    grade_front: list[float]
    speed_meters_per_second: list[float]
    speed_limit_meters_per_second: list[float]
    speed_target_meters_per_second: list[float]
    dt_seconds: list[float]
    length_meters: list[float]
    mass_static_kilograms: list[float]
    mass_adj_kilograms: list[float]
    mass_freight_kilograms: list[float]
    max_fric_braking: list[float]
    weight_static_newtons: list[float]
    res_rolling_newtons: list[float]
    res_bearing_newtons: list[float]
    res_davis_b_newtons: list[float]
    res_aero_newtons: list[float]
    res_grade_newtons: list[float]
    res_curve_newtons: list[float]
    pwr_whl_out_watts: list[float]
    energy_whl_out_joules: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

class LinkPoint(SerdeAPI):
    offset_meters: float
    grade_count: int
    curve_count: int
    cat_power_count: int
    link_idx: LinkIdx

class PathResCoeff(SerdeAPI):
    offset: float
    res_coeff: float
    res_net: float

class SpeedLimitPoint(SerdeAPI):
    offset: float
    speed_limit: float

class CatPowerLimit(SerdeAPI):
    offset_start: float
    offset_end: float
    power_limit: float
    district_id: str | None

class TrainParams(SerdeAPI):
    length: float
    speed_max: float
    mass_static: float
    mass_per_brake: float
    axle_count: int
    train_type: str
    curve_coeff_0: float
    curve_coeff_1: float
    curve_coeff_2: float

class PathTpc(SerdeAPI):
    link_points: list[LinkPoint]
    grades: list[PathResCoeff]
    curves: list[PathResCoeff]
    speed_points: list[SpeedLimitPoint]
    cat_power_limits: list[CatPowerLimit]
    train_params: TrainParams
    is_finished: bool

class BrakingPoint(SerdeAPI):
    offset_meters: float
    speed_limit_meters_per_second: float
    speed_target_meters_per_second: float

class BrakingPoints(SerdeAPI):
    points: list[BrakingPoint]
    idx_curr: int

class FricBrakeState(SerdeAPI):
    i: int
    force_newtons: float
    force_max_curr_newtons: float

class FricBrakeStateHistoryVec(SerdeAPI):
    i: list[int]
    force_newtons: list[float]
    force_max_curr_newtons: list[float]

class FricBrake(SerdeAPI):
    force_max_newtons: float
    ramp_up_time_seconds: float
    ramp_up_coeff_ratio: float
    state: FricBrakeState
    history: FricBrakeStateHistoryVec
    save_interval: int | None

class SpeedLimitTrainSim(SerdeAPI):
    train_id: str
    origs: list[Location]
    dests: list[Location]
    loco_con: Consist
    n_cars_by_type: dict[str, int]
    state: TrainState
    # train_res: TrainRes # not accessible in Python
    path_tpc: PathTpc
    braking_points: BrakingPoints
    fric_brake: FricBrake
    history: TrainStateHistoryVec
    save_interval: int | None
    simulation_days: int | None
    scenario_year: int | None

    @classmethod
    def __init__(
        cls,
        loco_con: Consist,
        n_cars_by_type: dict[str, int],
        state: TrainState,
        train_res_file: str | None,
        path_tpc_file: str | None,
        speed_trace: SpeedTrace,
        save_interval: int | None,
        simulation_days: int | None,
    ) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def set_save_interval(self, save_interval: int): ...
    def walk(self): ...
    def walk_timed_path(self, network: Network, timed_path: list[LinkIdxTime]): ...
    def get_energy_fuel_joules(self, soc_correction: bool = True) -> float: ...
    def get_energy_fuel_soc_corrected_joules(self) -> float: ...
    def to_dataframe(self) -> Any: ...

@dataclass
class SpeedLimitTrainSimVec(SerdeAPI):
    speed_limit_train_sims: list[SpeedLimitTrainSim]
    @classmethod
    def default(cls) -> Self: ...
    def tolist(self) -> list[SpeedLimitTrainSim]: ...
    def set_save_interval(self, save_interval: int): ...

@dataclass
class LinkIdx(SerdeAPI):
    idx: int
    @classmethod
    def default(cls) -> Self: ...

class LinkIdxTime(SerdeAPI):
    link_idx: LinkIdx
    time_seconds: float
    @classmethod
    def default(cls) -> Self: ...

class TrainSimBuilder(SerdeAPI):
    train_id: str
    origin_id: str
    destination_id: str
    train_config: TrainConfig
    loco_con: Consist
    init_train_state: InitTrainState | None
    drag_coeff_vec: list[float] | None

    @classmethod
    def default(cls) -> Self: ...
    def __init__(
        self,
        train_id: str | None = None,
        origin_id: str | None = None,
        destination_id: str | None = None,
        train_config: TrainConfig | None = None,
        loco_con: Consist | None = None,
        init_train_state: InitTrainState | None = None,
    ) -> None: ...
    def make_set_speed_train_sim(
        self,
        rail_vehicles: list[RailVehicle] | None = None,
        network: list[Link] | Network | None = None,
        link_path: list[LinkIdx] | LinkPath | None = None,
        speed_trace: SpeedTrace | None = None,
        save_interval: int | None = None,
        temp_trace: TemperatureTrace | None = None,
    ) -> SetSpeedTrainSim: ...
    def make_speed_limit_train_sim(
        self,
        rail_vehicles: list[RailVehicle] | None = None,
        location_map: dict[str, list[Location]] | None = None,
        save_interval: int | None = None,
        simulation_days: int | None = None,
        scenario_year: int | None = None,
        temp_trace: TemperatureTrace | None = None,
    ) -> SpeedLimitTrainSim: ...

@dataclass
class TrainConfig(SerdeAPI):
    n_cars_by_type: dict[str, int] | None = None
    rail_vehicle_type: str | None = None
    rail_vehicles: list[RailVehicle] | None = None
    train_type: str | None = None
    train_length_meters: float | None = None
    train_mass_kilograms: float | None = None
    cd_area_vec: list[float] | None = None

    def __init__(
        self,
        n_cars_by_type: dict[str, int] | None = None,
        rail_vehicles: list[RailVehicle] | None = None,
        train_length_meters: float | None = None,
        train_mass_kilograms: float | None = None,
        **kwargs: Any,
    ) -> None: ...

    @classmethod
    def default(cls) -> Self: ...

class RailVehicle(SerdeAPI):
    axle_count: int
    bearing_res_per_axle_newtons: float
    brake_count: int
    braking_ratio_empty: float
    braking_ratio_loaded: float
    car_type: str
    freight_type: str
    davis_b_seconds_per_meter: float
    cd_area_empty_square_meters: float
    cd_area_loaded_square_meters: float
    length_meters: float
    mass_rot_per_axle_kilograms: float
    mass_static_empty_kilograms: float
    mass_static_loaded_kilograms: float
    rolling_ratio: float
    speed_max_empty_meters_per_second: float
    speed_max_loaded_meters_per_second: float

    @classmethod
    def default(cls) -> Self: ...

# Additional missing classes referenced in the demos
class Network(SerdeAPI):
    @classmethod
    def default(cls) -> Self: ...

class Location(SerdeAPI):
    @classmethod
    def default(cls) -> Self: ...

class Link(SerdeAPI):
    @classmethod
    def default(cls) -> Self: ...

class LinkPath(SerdeAPI):
    def __init__(self, link_indices: list[LinkIdx]) -> None: ...
    @classmethod
    def default(cls) -> Self: ...
    @classmethod
    def from_csv_file(cls, pathstr: str | Path) -> Self: ...
    def to_csv_file(self, pathstr: str | Path) -> None: ...

class TimedLinkPath(SerdeAPI):
    @classmethod
    def default(cls) -> Self: ...

class TemperatureTrace(SerdeAPI):
    @classmethod
    def default(cls) -> Self: ...

class InitTrainState(SerdeAPI):
    @classmethod
    def default(cls) -> Self: ...

# Module-level functions
def import_locations(file_path: str | Path) -> dict[str, list[Location]]: ...
def resources_root() -> Path: ...
def make_est_times(train_sim: Any, network: Network) -> tuple[Any, Any]: ...
def run_dispatch(
    network: Network,
    train_sims: SpeedLimitTrainSimVec,
    est_time_nets: list[Any],
    debug1: bool,
    debug2: bool,
) -> tuple[Any, Any]: ...
def run_speed_limit_train_sims(
    speed_limit_train_sims: SpeedLimitTrainSim,
    network: list[Link],
    train_consist_plan_py: Any,
    loco_pool_py: Any,
    refuel_facilities_py: Any,
    timed_paths: list[TimedLinkPath],
) -> tuple[list[Any], list[Any]]: ...
def simulate_prescribed_rollout(
    network_filename_path: str,
    demand_file: str,
    simulation_days: int,
    scenario_year: int,
    **kwargs: Any,
) -> dict[str, Any]: ...
