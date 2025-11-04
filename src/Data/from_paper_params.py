import pybamm
import numpy as np

def get_samsung_25r_parameters():
    """
    Returns a parameter set for Samsung 25R 18650 cell based on
    reverse engineering measurements from Kartini et al. (2023).

    Key findings from paper:
    - Cathode: NMC111 (LiNi1/3Mn1/3Co1/3O2) with good crystallinity
    - Particle size: 300-500 nm average (after cycling: 200-400 nm)
    - Layered hexagonal structure with high ordering
    - Nominal capacity: 2500 mAh
    """

    # Start with Chen2020 as base (NMC chemistry)
    params = pybamm.ParameterValues("Chen2020")

    # ============================================================================
    # GEOMETRIC PARAMETERS (Table 1 from paper)
    # ============================================================================

    # Note: Paper has typo - 18650 means 18mm diameter × 65mm length, not 650mm
    cell_diameter = 0.018  # m (18 mm)
    cell_length = 0.065  # m (65 mm, corrected from paper's 650 mm typo)

    # Cathode (positive electrode) - unrolled dimensions from Table 1
    cathode_length = 0.875  # m (875 mm)
    cathode_width = 0.057  # m (57 mm from Table 1, not 570 mm)
    cathode_total_thickness = 1.41e-3  # m (1.41 mm total)

    # Anode (negative electrode) - unrolled dimensions from Table 1
    anode_length = 0.930  # m (930 mm)
    anode_width = 0.057  # m (57 mm from Table 1)
    anode_total_thickness = 1.28e-3  # m (1.28 mm total)

    # Separator - from Table 1
    separator_length = 1.820  # m (1820 mm - twice electrode length)
    separator_width = 0.060  # m (60 mm - slightly wider to prevent short circuit)
    separator_thickness = 0.15e-3  # m (0.15 mm)

    # Current collectors (typical values for Li-ion, not specified in paper)
    al_foil_thickness = 16e-6  # m (16 μm typical for cathode)
    cu_foil_thickness = 10e-6  # m (10 μm typical for anode)

    # Calculate coating thicknesses
    # Assuming double-sided coating
    cathode_coating_total = cathode_total_thickness - al_foil_thickness
    cathode_thickness = cathode_coating_total / 2  # Each side

    anode_coating_total = anode_total_thickness - cu_foil_thickness
    anode_thickness = anode_coating_total / 2  # Each side

    # Electrode areas
    electrode_height = cathode_width  # 0.057 m
    cathode_area = cathode_length * cathode_width  # m²
    anode_area = anode_length * anode_width  # m²

    # ============================================================================
    # MATERIAL PARAMETERS (from XRD and SEM analysis)
    # ============================================================================

    # Particle sizes from SEM (Figure 4, page 5-6)
    # Average particle size: 300-500 nm (fresh), 200-400 nm (after cycling)
    positive_particle_radius = 400e-9  # m (400 nm - middle of range)

    # Anode particle size (not specified in paper, using typical graphite value)
    negative_particle_radius = 10e-6  # m (10 μm typical for graphite)

    # Crystallographic data from XRD (Table 2)
    # Lattice parameters for NMC111:
    # a = 2.8606 Å, c = 14.2687 Å
    # c/a = 4.9880 (> 4.899 ideal, indicating good ordering)
    # Crystallite size = 27913 Å = 2791.3 nm

    # ============================================================================
    # CHEMISTRY-SPECIFIC PARAMETERS
    # ============================================================================

    # The paper confirms NMC111 cathode (LiNi1/3Mn1/3Co1/3O2)
    # Update chemistry-specific parameters if needed
    # (Chen2020 uses NMC with similar composition)

    # ============================================================================
    # UPDATE PARAMETERS
    # ============================================================================

    params.update({
        # Geometric parameters
        "Electrode height [m]": electrode_height,
        "Electrode width [m]": cathode_length,
        "Negative electrode thickness [m]": anode_thickness,
        "Positive electrode thickness [m]": cathode_thickness,
        "Separator thickness [m]": separator_thickness,

        # Current collector thicknesses
        "Negative current collector thickness [m]": cu_foil_thickness,
        "Positive current collector thickness [m]": al_foil_thickness,

        # Particle radii
        "Negative particle radius [m]": negative_particle_radius,
        "Positive particle radius [m]": positive_particle_radius,

        # Cell capacity - Samsung 25R is rated at 2500 mAh
        "Nominal cell capacity [A.h]": 2.5,

        # Number of electrode pairs (single jelly roll in 18650)
        # "Number of electrode pairs connected in parallel to make a cell": 1,

        # Cell geometric properties
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-6,  # Typical for 18650

    })

    # ============================================================================
    # PRINT SUMMARY
    # ============================================================================

    print("=" * 70)
    print("Samsung 25R 18650 Parameter Set")
    print("Based on: Kartini et al. (2023), AIP Conf. Proc. 2932, 020012")
    print("=" * 70)

    print("\nCell Dimensions:")
    print(f"  18650 format: {cell_diameter * 1000:.0f} mm diameter × {cell_length:.0f} mm length")

    print("\nGeometric Parameters (unrolled):")
    print(f"  Cathode: {cathode_length * 1000:.0f} × {cathode_width * 1000:.0f} mm")
    print(f"  Anode:   {anode_length * 1000:.0f} × {anode_width * 1000:.0f} mm")
    print(f"  Separator: {separator_length * 1000:.0f} × {separator_width * 1000:.0f} mm")
    print(f"\n  Cathode area: {cathode_area * 1e4:.2f} cm²")
    print(f"  Anode area:   {anode_area * 1e4:.2f} cm²")

    print("\nThicknesses:")
    print(f"  Cathode coating (per side): {cathode_thickness * 1e6:.1f} μm")
    print(f"  Anode coating (per side):   {anode_thickness * 1e6:.1f} μm")
    print(f"  Separator: {separator_thickness * 1e6:.0f} μm")
    print(f"  Al current collector: {al_foil_thickness * 1e6:.0f} μm")
    print(f"  Cu current collector: {cu_foil_thickness * 1e6:.0f} μm")

    print("\nMaterial Properties:")
    print(f"  Cathode: NMC111 (LiNi₁/₃Mn₁/₃Co₁/₃O₂)")
    print(f"  Cathode particle size: {positive_particle_radius * 1e9:.0f} nm")
    print(f"  Anode particle size: {negative_particle_radius * 1e6:.0f} μm (typical graphite)")
    print(f"  XRD crystallite size: 2791 nm (highly crystalline)")
    print(f"  c/a ratio: 4.988 (good layered ordering)")

    print("\nElectrochemical:")
    print(f"  Nominal capacity: 2500 mAh")
    print(f"  Nominal voltage: ~3.6 V (typical for NMC111)")

    print("=" * 70)
    print("\nNote: Current collector thicknesses are typical values.")
    print("Paper provides total electrode thickness including coating + foil.")
    print("=" * 70)

    return params


def get_samsung_25r_parameters_v7():
    """
    Returns a parameter set for Samsung 25R 18650 cell based on
    reverse engineering measurements from Kartini et al. (2023).

    Key findings from paper:
    - Cathode: NMC111 (LiNi1/3Mn1/3Co1/3O2) with good crystallinity
    - Particle size: 300-500 nm average (after cycling: 200-400 nm)
    - Layered hexagonal structure with high ordering
    - Nominal capacity: 2500 mAh
    """

    # Start with Chen2020 as base (NMC chemistry)
    params = pybamm.ParameterValues("Chen2020")

    # ============================================================================
    # GEOMETRIC PARAMETERS (Table 1 from paper)
    # ============================================================================

    # Note: Paper has typo - 18650 means 18mm diameter × 65mm length, not 650mm
    cell_diameter = 0.018  # m (18 mm)
    cell_length = 0.065  # m (65 mm, corrected from paper's 650 mm typo)

    # Cathode (positive electrode) - unrolled dimensions from Table 1
    # NOTE: Table 1 shows width as 57mm, but text says 570mm - using 57mm from table
    cathode_length = 0.875  # m (875 mm)
    cathode_width = 0.057  # m (57 mm from Table 1)

    cathode_total_thickness = 1.41e-3  # m (1.41 mm total including foil)

    # Anode (negative electrode) - unrolled dimensions from Table 1
    anode_length = 0.930  # m (930 mm)
    anode_width = 0.057  # m (57 mm from Table 1)
    anode_total_thickness = 1.28e-3  # m (1.28 mm total including foil)

    # Separator - from Table 1
    separator_length = 1.820  # m (1820 mm - twice electrode length)
    separator_width = 0.060  # m (60 mm - slightly wider to prevent short circuit)
    separator_thickness = 0.15e-3  # m (0.15 mm)



    print(f"\nDEBUG - Calculated coating thicknesses:")
    # print(f"  Cathode per side: {cathode_thickness * 1e6:.1f} μm")
    # print(f"  Anode per side: {anode_thickness * 1e6:.1f} μm")
    print(f"  WARNING: If these are >200 μm, the paper's measurements may be")
    print(f"           for the entire rolled stack, not single electrode thickness!")

    # Electrode areas
    electrode_height = cathode_width  # 0.057 m
    cathode_area = cathode_length * cathode_width  # m²
    anode_area = anode_length * anode_width  # m²

    # ============================================================================
    # MATERIAL PARAMETERS (from XRD and SEM analysis)
    # ============================================================================

    # Particle sizes from SEM (Figure 4, page 5-6)
    # Average particle size: 300-500 nm (fresh), 200-400 nm (after cycling)

    positive_particle_radius = 5e-7


    # ============================================================================
    # CHEMISTRY-SPECIFIC PARAMETERS
    # ============================================================================

    # The paper confirms NMC111 cathode (LiNi1/3Mn1/3Co1/3O2)
    # Update chemistry-specific parameters if needed
    # (Chen2020 uses NMC with similar composition)

    # ============================================================================
    # UPDATE PARAMETERS
    # ============================================================================

    params.update({
        # Geometric parameters
        "Electrode height [m]": electrode_height,
        "Electrode width [m]": cathode_length,

        "Separator thickness [m]": separator_thickness,


        "Positive particle radius [m]": positive_particle_radius,

        # # Cell capacity - Samsung 25R is rated at 2500 mAh
        "Nominal cell capacity [A.h]": 2.5,
        #
        # # Number of electrode pairs (single jelly roll in 18650)
        # # "Number of electrode pairs connected in parallel to make a cell": 1,
        #
        # # Cell geometric properties
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-6,
        #

    }, check_already_exists=True)

    # ============================================================================
    # PRINT SUMMARY
    # ============================================================================

    print("=" * 70)
    print("Samsung 25R 18650 Parameter Set")
    print("Based on: Kartini et al. (2023), AIP Conf. Proc. 2932, 020012")
    print("=" * 70)

    print("\nCell Dimensions:")
    print(f"  18650 format: {cell_diameter * 1000:.0f} mm diameter × {cell_length:.0f} mm length")

    print("\nGeometric Parameters (unrolled):")
    print(f"  Cathode: {cathode_length * 1000:.0f} × {cathode_width * 1000:.0f} mm")
    print(f"  Anode:   {anode_length * 1000:.0f} × {anode_width * 1000:.0f} mm")
    print(f"  Separator: {separator_length * 1000:.0f} × {separator_width * 1000:.0f} mm")
    print(f"\n  Cathode area: {cathode_area * 1e4:.2f} cm²")
    print(f"  Anode area:   {anode_area * 1e4:.2f} cm²")

    print("\nThicknesses:")
    print(f"  Separator: {separator_thickness * 1e6:.0f} μm")


    print("\nMaterial Properties:")
    print(f"  Cathode: NMC111 (LiNi₁/₃Mn₁/₃Co₁/₃O₂)")
    print(f"  Cathode particle size: {positive_particle_radius * 1e9:.0f} nm")

    print("=" * 70)
    print("\nNote: Current collector thicknesses are typical values.")
    print("Paper provides total electrode thickness including coating + foil.")
    print("=" * 70)

    return params