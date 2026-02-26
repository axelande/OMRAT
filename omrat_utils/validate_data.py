from pydantic import BaseModel, Field, RootModel
from typing import Dict, List, Optional, Union

class PC(BaseModel):
    p_pc: float
    d_pc: float
    headon: float = 4.9E-5
    overtaking: float = 1.1E-4
    crossing: float = 1.3E-4
    bend: float = 1.3E-4
    grounding: float = 1.6E-4
    allision: float = 1.9E-4

class Rose(RootModel[Dict[str, float]]):
    pass  # keys like "0", "45", etc.

class Repair(BaseModel):
    func: str
    std: float
    loc: float
    scale: float
    use_lognormal: bool

class Drift(BaseModel):
    drift_p: int
    anchor_p: float
    anchor_d: int
    speed: float
    rose: Rose
    repair: Repair

class TrafficDirectionData(BaseModel):
    Frequency_ships_per_year: List[List[float]] = Field(alias="Frequency (ships/year)")
    Speed_knots: List[List[float]] = Field(alias="Speed (knots)")
    Draught_meters: List[List[float]] = Field(alias="Draught (meters)")
    Ship_heights_meters: List[List[float]] = Field(alias="Ship heights (meters)")
    Ship_Beam_meters: List[List[float]] = Field(alias="Ship Beam (meters)")

class TrafficLeg(BaseModel):
    East_going: TrafficDirectionData = Field(alias="East going")
    West_going: TrafficDirectionData = Field(alias="West going")

class TrafficData(RootModel[Dict[str, TrafficLeg]]):
    pass

class Segment(BaseModel):
    Start_Point: str
    End_Point: str
    Dirs: List[str]
    Width: int
    line_length: float
    Route_Id: int
    Leg_name: str
    Segment_Id: str
    mean1_1: float
    std1_1: float
    mean2_1: float
    std2_1: float
    weight1_1: float
    weight2_1: float
    mean1_2: float
    mean1_3: float
    std1_2: float
    std1_3: float
    mean2_2: float
    mean2_3: float
    std2_2: float
    std2_3: float
    weight1_2: float
    weight1_3: float
    weight2_2: float
    weight2_3: float
    u_min1: float
    u_max1: float
    u_p1: int
    ai1: float
    u_min2: float
    u_max2: float
    u_p2: int
    ai2: float

class SegmentData(RootModel[Dict[str, Segment]]):
    pass

class PolygonEntry(RootModel[List[str]]):
    pass  # ["1", "0.0-3.0", "MultiPolygon (...)"]

class Depths(RootModel[List[PolygonEntry]]):
    pass

class Objects(RootModel[List[PolygonEntry]]):
    pass

class LengthInterval(BaseModel):
    min: Union[float, str]
    max: Union[float, str]
    label: str

class ShipCategoriesModel(BaseModel):
    types: List[str]
    length_intervals: List[LengthInterval]
    selection_mode: Optional[str] = None

class RootModelSchema(BaseModel):
    pc: PC
    drift: Drift
    traffic_data: TrafficData
    segment_data: SegmentData
    depths: Depths
    objects: Objects
    ship_categories: Optional[ShipCategoriesModel] = None