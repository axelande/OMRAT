from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import MagicMock, PropertyMock, patch
from omrat_utils.handle_ais import AIS
from omrat_utils.handle_traffic import Traffic 
from omrat_utils.handle_ship_cat import ShipCategories

# filepath: omrat_utils/test_handle_ais.py

@pytest.fixture
def mock_ais():
    with patch("omrat_utils.handle_ais.DB") as MockDB:
        mock_db_instance = MagicMock()
        MockDB.return_value = mock_db_instance

        mock_omrat = MagicMock()
        sc = ShipCategories(MagicMock())
        mock_omrat.ship_cat = sc
        mock_omrat.traffic_data = {}
        ais = AIS(mock_omrat)
        
        # Optionally, set up additional mocks
        mock_traffic = Traffic(mock_omrat, MagicMock())
        mock_omrat.traffic = mock_traffic
        ais.db = mock_db_instance  # Ensure .db is the mock
        ais.schema = "test_schema"
        ais.year = 2023
        ais.months = []
        return ais

dt = datetime(2024, 2, 14, 3, 47, 25, tzinfo=timezone(timedelta(seconds=3600)))
ais_return = [(87.0, 12.0, 79, 5.7, 'General Cargo Ship', dt, 8.6, 23.0, -2361.856017700104, 36), 
 (158.0, 23.0, 79, 7.7, 'container ship _fully cellular_', dt, 12.6, 0.0, -2379.829346660104, 36), 
 (219.0, 31.0, 69, 7.1, 'Passenger/Ro-Ro Ship (Vehicles)', dt, 21.4, 0.0, -1738.2160136101043, 37), 
 (90.0, 15.0, 75, 4.7, None, dt, 10.8, None, 1621.059160829896, 218), 
 (88.0, 13.0, 80, 6.7, 'chemical_products tanker', dt, 10.7, 31.4, -2260.559472040104, 40), 
 (90.0, 15.0, 70, 3.5, 'general cargo ship', dt, 11.4, 31.5, 1447.521590139896, 216), 
 (99.0, 16.0, 33, 6.5, 'Trailing Suction Hopper Dredger', dt, 10.4, 0.0, 1766.707647529896, 219), 
 (89.0, 16.0, 70, 5.4, None, dt, 10.1, None, -2450.674192640104, 36), 
 (190.0, 32.0, 70, 12.9, None, dt, 11.4, None, 1888.954612899896, 218), 
 (180.0, 28.0, 70, 10.4, None, dt, 10.7, None, 1417.2676958698958, 219), 
 (170.0, 28.0, 70, 6.0, None, dt, 11.0, None, -2244.7818205601043, 39), 
 (205.0, 26.0, 70, 7.9, 'Ro-Ro Cargo Ship', dt, 21.1, 0.0, 1670.6070451798955, 218), 
 (219.0, 31.0, 69, 7.1, 'Passenger/Ro-Ro Ship (Vehicles)', dt, 22.2, 0.0, 1607.1049918798958, 216), 
 (134.0, 22.0, 79, 7.5, 'general cargo ship', dt, 12.8, 0.0, 2370.6740117098957, 218), 
 (190.0, 32.0, 70, 11.8, None, dt, 11.1, None, 1302.360755989896, 217), 
 (87.0, 12.0, 79, 3.6, 'General Cargo Ship', dt, 7.9, 23.0, 1627.2530941898963, 215), 
 (180.0, 32.0, 70, 7.9, 'bulk carrier', dt, 13.1, 43.0, 2345.187432549896, 216), 
 (90.0, 15.0, 79, 5.9, 'general cargo ship', dt, 10.4, 0.0, 2018.7695683198958, 224), 
 (230.0, 38.0, 70, 7.0, None, dt, 13.1, None, -2110.0046351401043, 38), 
 (192.0, 26.0, 70, 6.9, 'ro_ro cargo ship', dt, 18.9, 0.0, -2374.428336830104, 40), 
 (90.0, 13.0, 79, 3.7, 'General Cargo Ship', dt, 11.9, 0.0, -1751.586262340104, 39), 
 (179.0, 28.0, 70, 11.0, None, dt, 12.3, None, 2186.518461749896, 220), 
 (189.0, 32.0, 70, 10.0, 'Bulk Carrier', dt, 10.6, 49.23, 2377.448895469896, 218), 
 (187.0, 27.0, 71, 6.7, 'Ro-Ro Cargo Ship', dt, 19.3, 0.0, -2237.9228519101043, 36), 
 (138.0, 21.0, 70, 6.5, 'General Cargo Ship', dt, 9.9, 43.04, 1421.1113377198958, 221), 
 (219.0, 31.0, 69, 7.1, 'Passenger/Ro-Ro Ship (Vehicles)', dt, 19.6, 0.0, 1496.4129969098958, 221), 
 (184.0, 28.0, 80, 10.8, None, dt, 12.0, None, 2356.5352839998955, 217), 
 (238.0, 35.0, 71, 7.4, None, dt, 18.9, None, 1245.5481197398958, 217),
 (90.0, 15.0, 70, 5.8, None, dt, 11.0, None, 1965.424991579896, 218), 
]


def test_run_sql_no_months(mock_ais: AIS):
    """Test run_sql when no months are selected."""
    mock_ais.months = []
    mock_ais.db.execute_and_return.return_value = [
        (100, 20, "type_and_cargo", 5.0, "ship_type", "2023-01-01", 15.0, 10.0, 100.0, 180.0)
    ]
    pl = "LINESTRING(10 20, 30 40)"
    result = mock_ais.run_sql(pl)

    # Verify the SQL query structure
    mock_ais.db.execute_and_return.assert_called_once()
    query = mock_ais.db.execute_and_return.call_args[0][0]
    assert "FROM test_schema.segments_2023" in query
    assert "ST_intersects(segment, ST_geomfromtext('LINESTRING(10 20, 30 40)', 4326))" in query

    # Verify the returned data
    assert result == [
        (100, 20, "type_and_cargo", 5.0, "ship_type", "2023-01-01", 15.0, 10.0, 100.0, 180.0)
    ]

def test_run_sql_with_months(mock_ais):
    """Test run_sql when specific months are selected."""
    mock_ais.months = [1, 2]
    mock_ais.db.execute_and_return.return_value = [
        (200, 25, "type_and_cargo", 6.0, "ship_type", "2023-02-01", 16.0, 11.0, 200.0, 190.0)
    ]
    pl = "LINESTRING(50 60, 70 80)"
    result = mock_ais.run_sql(pl)

    # Verify the SQL query structure
    mock_ais.db.execute_and_return.assert_called_once()
    query = mock_ais.db.execute_and_return.call_args[0][0]
    assert "FROM test_schema.segments_2023_1" in query
    assert "FROM test_schema.segments_2023_2" in query
    assert "ST_intersects(segment, ST_geomfromtext('LINESTRING(50 60, 70 80)', 4326))" in query

    # Verify the returned data
    assert result == [
        (200, 25, "type_and_cargo", 6.0, "ship_type", "2023-02-01", 16.0, 11.0, 200.0, 190.0)
    ]

def test_run_sql_empty_result(mock_ais):
    """Test run_sql when the query returns no results."""
    mock_ais.db.execute_and_return.return_value = []
    pl = "LINESTRING(0 0, 10 10)"
    result = mock_ais.run_sql(pl)

    # Verify the SQL query structure
    mock_ais.db.execute_and_return.assert_called_once()
    query = mock_ais.db.execute_and_return.call_args[0][0]
    assert "ST_intersects(segment, ST_geomfromtext('LINESTRING(0 0, 10 10)', 4326))" in query

    # Verify the returned data
    assert result == []
    
@pytest.fixture
def mock_table():
    """Fixture to mock QTableWidget."""
    table = MagicMock()
    table.rowCount.return_value = 1
    table.item.side_effect = lambda row, col: MagicMock(text=PropertyMock(return_value=[
        '1', '1', '14.33188 55.21143', '14.52057 55.35013', '5000'
    ][col]))
    return table

def test_get_segment_data_from_table_valid_data(mock_ais, mock_table):
    """Test get_segment_data_from_table with valid data."""
    mock_ais.omrat.dockwidget.twRouteList = mock_table
    result = mock_ais.get_segment_data_from_table()

    # Verify the result
    assert result == {
        '1': {
            'Route Id': '1',
            'Start Point': '14.33188 55.21143',
            'End Point': '14.52057 55.35013',
            'Width': 5000.0
        }
    }

def test_get_segment_data_from_table_missing_data(mock_ais, mock_table):
    """Test get_segment_data_from_table with missing data."""
    # Simulate missing data in the table
    mock_table.item.side_effect = lambda row, col: MagicMock(text=PropertyMock(return_value=[
        '1', None, '14.33188 55.21143', '14.52057 55.35013', '5000'
    ][col]))
    mock_ais.omrat.dockwidget.twRouteList = mock_table
    result = mock_ais.get_segment_data_from_table()

    # Verify the result is empty due to missing data
    assert result == {}

def test_update_legs(mock_ais, mock_table):
    """Test update_legs with mocked segment data."""
    # Mock the QTableWidget
    mock_ais.omrat.dockwidget.twRouteList = mock_table

    # Mock other dependencies
    mock_ais.omrat.qgis_geoms.leg_dirs = {'1': ['East going', 'West going']}
    mock_ais.omrat.traffic.run_update_plot = MagicMock()
    mock_ais.db.execute_and_return = MagicMock(side_effect=[
        [['LINESTRING(14.435042123303658 55.24939672092081,14.372456499980117 55.276595228118936)']],
        ais_return,# Mock ais_data query result
        [[37.0]],  # Mock leg bearing query result
        
    ])

    # Call update_legs
    mock_ais.update_legs()

    # Verify that run_sql was called and returned ais_return
    assert mock_ais.db.execute_and_return.call_count == 3
    assert mock_ais.db.execute_and_return.call_args_list[1][0][0]  # Ensure the second call is for ais_data
