#include "gtest/gtest.h"
#define private public
#include "source_io/module_parameter/parameter.h"
#undef private
#include "source_cell/read_stru.h"
#include "source_cell/unitcell.h"
#define private public
#define protected public
#include "source_esolver/esolver_nep.h"
#include "source_md/md_func.h"
#include "source_base/global_variable.h"

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

/**
 * MD_func::force_virial with NEP (same HfO2 STRU + nep_hfo2 model as tests/04_FF/101_NEP_HfO2).
 * Built only when NEP_DIR is set (__NEP). ABACUS_SOURCE_ROOT must point to the repository root.
 */

#ifndef ABACUS_SOURCE_ROOT
#define ABACUS_SOURCE_ROOT ""
#endif

namespace
{
bool load_hf_o2_stru(UnitCell& ucell, const std::string& stru_path)
{
    std::ifstream ifa(stru_path.c_str());
    if (!ifa.is_open())
    {
        return false;
    }
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("/dev/null");
    ofs_warning.open("/dev/null");

    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "pw";
    PARAM.input.nspin = 1;
    PARAM.input.dfthalf_type = 0;

    ucell.ntype = 2;
    ucell.atoms = new Atom[ucell.ntype];
    ucell.set_atom_flag = true;

    if (!unitcell::read_atom_species(ifa, ofs_running, ucell))
    {
        ofs_running.close();
        ofs_warning.close();
        return false;
    }
    if (!unitcell::read_lattice_constant(ifa, ofs_running, ucell.lat))
    {
        ofs_running.close();
        ofs_warning.close();
        return false;
    }

    delete[] ucell.magnet.start_mag;
    ucell.magnet.start_mag = new double[ucell.ntype];
    if (!unitcell::read_atom_positions(ucell, ifa, ofs_running, ofs_warning))
    {
        ofs_running.close();
        ofs_warning.close();
        return false;
    }

    ofs_running.close();
    ofs_warning.close();
    ifa.close();

    assert(ucell.lat0 > 0.0);
    ucell.omega = ucell.latvec.Det() * ucell.lat0 * ucell.lat0 * ucell.lat0;
    if (ucell.omega < 0)
    {
        ucell.omega = std::abs(ucell.omega);
    }
    ucell.GT = ucell.latvec.Inverse();
    ucell.G = ucell.GT.Transpose();
    ucell.GGT = ucell.G * ucell.GT;
    ucell.invGGT = ucell.GGT.Inverse();
    ucell.GT0 = ucell.GT;
    ucell.G0 = ucell.G;
    ucell.GGT0 = ucell.GGT;
    ucell.invGGT0 = ucell.invGGT;
    ucell.set_iat2itia();
    return true;
}
} // namespace

#ifdef __NEP

class MD_nep_mdfunc_Test : public ::testing::Test
{
  protected:
    UnitCell ucell;
    Input_para inp;
    ModuleBase::Vector3<double>* force = nullptr;
    ModuleBase::matrix virial;
    ModuleBase::matrix stress;
    double potential = 0.0;

    void SetUp() override
    {
        const std::string root = std::string(ABACUS_SOURCE_ROOT);
        if (root.empty())
        {
            GTEST_SKIP() << "ABACUS_SOURCE_ROOT not defined.";
        }
        PARAM.sys.global_out_dir = "./";
        PARAM.sys.global_readin_dir = "./";
    PARAM.inp.pseudo_dir = root + "/tests/PP_ORB";
    PARAM.inp.press1 = PARAM.inp.press2 = PARAM.inp.press3 = 0.0;
    GlobalV::ofs_running.open("/dev/null");

        const std::string stru = root + "/tests/04_FF/101_NEP_HfO2/STRU";
        if (!load_hf_o2_stru(ucell, stru))
        {
            GTEST_SKIP() << "Cannot read STRU at " << stru;
        }

        force = new ModuleBase::Vector3<double>[ucell.nat];
        virial.create(3, 3);
        stress.create(3, 3);
    }

    void TearDown() override
    {
        delete[] force;
        GlobalV::ofs_running.close();
    }
};

TEST_F(MD_nep_mdfunc_Test, force_virial_nep_matches_reference_energy)
{
    const std::string root = std::string(ABACUS_SOURCE_ROOT);
    const std::string model = root + "/tests/PP_ORB/nep_hfo2.txt";

    ModuleESolver::ESolver* p_esolver = new ModuleESolver::ESolver_NEP(model);
    p_esolver->before_all_runners(ucell, inp);

    MD_func::force_virial(p_esolver, 0, ucell, potential, force, true, virial);

    std::vector<ModuleBase::Vector3<double>> vel0(ucell.nat);
    std::vector<double> mass(ucell.nat, 1.0);
    MD_func::compute_stress(ucell, vel0.data(), mass.data(), true, virial, stress);

    EXPECT_TRUE(std::isfinite(potential));
    for (int i = 0; i < ucell.nat; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            EXPECT_TRUE(std::isfinite(force[i][k]));
        }
    }

    // tests/04_FF/101_NEP_HfO2/result.ref etotref in Ry; MD_func stores Hartree (Ry * 0.5).
    const double etot_ry = -243.9772424704458;
    EXPECT_NEAR(potential, 0.5 * etot_ry, 5.0e-2);

    delete p_esolver;
}

#else

TEST(MD_nep_mdfunc_placeholder, skipped_without_nep)
{
    GTEST_SKIP() << "Build with -DNEP_DIR=... to enable NEP MD_func tests.";
}

#endif
