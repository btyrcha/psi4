import pytest
import psi4
from qcelemental import constants
from psi4 import compare_values

hartree_to_kcalmol = constants.conversion_factor("hartree", "kcal/mol")
pytestmark = [pytest.mark.psi, pytest.mark.api]

_sapt_testing_mols = {
    "neutral_water_dimer": """
0 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
1   0.257521062   0.042121496   0.005218999
--
0 1
8   2.268880784   0.026340101   0.000508029
1   2.645502399   -0.412039965   0.766632411
1   2.641145101   -0.449872874   -0.744894473
units angstrom
""",
    "hydroxide": """
-1 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
--
0 1
8   2.268880784   0.026340101   0.000508029
1   2.645502399   -0.412039965   0.766632411
1   2.641145101   -0.449872874   -0.744894473
units angstrom
""",
    "he_h2": """
symmetry c1
no_com
no_reorient
units bohr
0 1
He  -2.500000000   0.000000000   0.000000000
--
0 1
H    1.775500000   0.000000000   0.000000000
H    3.224500000   0.000000000   0.000000000
""",
}


@pytest.mark.saptdft
@pytest.mark.parametrize(
    "SAPT_DFT_GRAC_COMPUTE, refA, refB, gracA, gracB, geometry, grac_basis",
    [
        (
            "SINGLE",
            0.19807358,
            0.19830016,
            None,
            None,
            'neutral_water_dimer',
            None,
        ),
        (
            "SINGLE",
            0.1307,
            0.19830016,
            0.1307,
            None,
            'neutral_water_dimer',
            None,
        ),
        (
            "ITERATIVE",
            0.2303073898,
            0.19830016,
            None,
            None,
            "hydroxide",
            None,
        ),
        (
            "ITERATIVE",
            0.0924377691,
            0.1306379832,
            None,
            None,
            "hydroxide",
            "aug-cc-pvdz",
        ),
    ],
)
def test_saptdft_auto_grac(SAPT_DFT_GRAC_COMPUTE, refA, refB, gracA, gracB, geometry, grac_basis):
    """
    For SAPT(DFT), one must compute a GRAC shift for each monomer. Ideally,
    this GRAC shift should be close to the experimental Ionization Potential
    (IP) of the monomer. Basis set incompleteness prevents this here.
    e.g., aug-DZ H2O has a shift of 0.1306, compared to 0.1307 experimental IP.
    """
    if psi4.core.get_option("scf", "integral_package") == "LIBCINT":
        grac_tol = 4.e-8
    else:
        grac_tol = 1.e-8
    mol_dimer = psi4.geometry(_sapt_testing_mols[geometry])
    psi4.set_options(
        {
            "basis": "STO-3G",
            "SAPT_DFT_FUNCTIONAL": "pbe0",
            "SAPT_DFT_GRAC_COMPUTE": SAPT_DFT_GRAC_COMPUTE,
        }
    )
    if grac_basis is not None:
        psi4.set_options({"SAPT_DFT_GRAC_BASIS": grac_basis})
    if gracA is not None:
        psi4.set_options({"SAPT_DFT_GRAC_SHIFT_A": gracA})
    if gracB is not None:
        psi4.set_options({"SAPT_DFT_GRAC_SHIFT_B": gracB})
    psi4.energy("SAPT(DFT)", molecule=mol_dimer)
    assert compare_values(
        refA,
        psi4.core.variable("SAPT DFT GRAC SHIFT A"),
        grac_tol,
        "SAPT DFT GRAC SHIFT A",
    )
    assert compare_values(
        refB,
        psi4.core.variable("SAPT DFT GRAC SHIFT B"),
        grac_tol,
        "SAPT DFT GRAC SHIFT B",
    )


@pytest.mark.saptdft
def test_saptdft_hybrid_small_monomer():
    """
    The hybrid xc kernel of the coupled FDDS dispersion QR-factorizes each monomer's (ar|Q), an
    (nocc*nvir) x naux matrix. Here both monomers have nocc*nvir = 26 < naux = 59, so only an
    economy QR (with min(nocc*nvir, naux) columns of Q) is defined; the full QR used before gave
    run-dependent garbage or NaN (#2178). The reference agrees, to about 1e-14 Eh in Disp20, with
    a dense evaluation of the same equations that uses pinv((ar|Q)^T) instead of a QR. With
    nocc*nvir < naux the hybrid kernel is not projected, so Disp20 also equals, to about 1e-14 Eh,
    the coupled response of Eq. (39) of Xie et al., JCP 157, 024801 (2022) evaluated densely.
    """
    psi4.geometry(_sapt_testing_mols["he_h2"])
    psi4.set_options({
        "basis": "aug-cc-pvdz",
        "scf_type": "direct",
        "e_convergence": 1.e-10,
        "d_convergence": 1.e-10,
        "SAPT_DFT_FUNCTIONAL": "pbe0",
        "SAPT_DFT_GRAC_SHIFT_A": 0.2433,
        "SAPT_DFT_GRAC_SHIFT_B": 0.1375,
        "SAPT_DFT_DO_DHF": False,
        "SAPT_DFT_DO_HYBRID": True,
    })
    psi4.energy("SAPT(DFT)")
    assert compare_values(-4.062469577229e-04, psi4.variable("SAPT DISP ENERGY"), 1.e-8, "SAPT DISP ENERGY")
    assert compare_values(6.092645712810e-04, psi4.variable("SAPT TOTAL ENERGY"), 1.e-8, "SAPT TOTAL ENERGY")


if __name__ == "__main__":
    test_saptdft_auto_grac(
            "ITERATIVE",
            0.0924377691,
            0.1306379832,
            None,
            None,
            "hydroxide",
            "aug-cc-pvdz",
        )

