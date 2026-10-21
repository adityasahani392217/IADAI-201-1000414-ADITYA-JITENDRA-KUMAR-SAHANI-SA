"""
Unit Tests - Geometric Polygon Operations & Point Validation
"""

import pytest
import numpy as np
from shapely.geometry import Polygon, box, Point


def test_polygon_area_computation():
    """Verify standard rectangle polygon area calculation."""
    poly = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
    assert poly.area == 5000.0


def test_ioa_full_overlap():
    """Verify IoA calculation when bounding box completely covers the slot."""
    slot_poly = Polygon([(10, 10), (50, 10), (50, 50), (10, 50)])  # Area = 1600
    veh_poly = box(0, 0, 60, 60)  # Completely covers slot
    
    intersection = slot_poly.intersection(veh_poly).area
    ioa = intersection / slot_poly.area
    assert ioa == 1.0


def test_ioa_partial_overlap():
    """Verify IoA calculation for partial overlap."""
    slot_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])  # Area = 10000
    veh_poly = box(50, 0, 150, 100)  # 50% overlap
    
    intersection = slot_poly.intersection(veh_poly).area
    ioa = intersection / slot_poly.area
    assert ioa == pytest.approx(0.5, 0.01)


def test_ground_contact_point():
    """Verify bottom-center ground contact point calculation."""
    # Bounding box: [x1, y1, x2, y2]
    bbox = [100, 200, 300, 400]
    expected_bc = (200.0, 400.0)
    
    bc_x = (bbox[0] + bbox[2]) / 2.0
    bc_y = float(bbox[3])
    
    assert (bc_x, bc_y) == expected_bc
    
    slot_poly = Polygon([(150, 350), (250, 350), (250, 450), (150, 450)])
    assert slot_poly.contains(Point(bc_x, bc_y))
