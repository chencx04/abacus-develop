#include "gtest/gtest.h"
#define private public
#include "source_io/module_parameter/parameter.h"
#undef private
#define private public
#define protected public
#include "source_esolver/esolver_dp.h"
#include "source_md/md_func.h"
#include "source_base/global_variable.h"

#include <cmath>
#include <fstream>
#include <vector>

/**
 * MD_func::force_virial with a real DeePMD model (frozen graph case_1.pb from esolver tests).
 * Built only when CMake is configured with DeePMD_DIR (__DPMD defined).
 */

namespace
{
void setup_minimal_dp_ucell(UnitCell& ucell, Input_para& inp)
{
    inp.mdp.dp_rescaling = 1.0;
    inp.mdp.dp_fparam.clear();
    inp.mdp.dp_aparam.clear();

    ucell.ntype = 2;
    ucell.nat = 2;
    ucell.atoms = new Atom[ucell.ntype];
    ucell.set_atom_flag = true;

    ucell.atom_label.resize(ucell.ntype);
    ucell.atom_mass.resize(ucell.ntype);
    ucell.atom_label[0] = "Cu";
    ucell.atom_label[1] = "Al";
    ucell.atom_mass[0] = 63.546;
    ucell.atom_mass[1] = 26.982;

    ucell.atoms[0].label = "Cu";
    ucell.atoms[1].label = "Al";
    ucell.atoms[0].na = 1;
    ucell.atoms[1].na = 1;
    ucell.atoms[0].mass = ucell.atom_mass[0];
    ucell.atoms[1].mass = ucell.atom_mass[1];

    for (int it = 0; it < ucell.ntype; ++it)
    {
        ucell.atoms[it].tau.resize(ucell.atoms[it].na);
        ucell.atoms[it].taud.resize(ucell.atoms[it].na);
        ucell.atoms[it].mbl.resize(ucell.atoms[it].na);
        ucell.atoms[it].vel.resize(ucell.atoms[it].na);
    }

    ucell.lat0 = 1.0;
    ucell.lat0_angstrom = ucell.lat0 * ModuleBase::BOHR_TO_A;
    ucell.latvec.e11 = ucell.latvec.e22 = ucell.latvec.e33 = 10.0;
    ucell.latvec.e12 = ucell.latvec.e13 = ucell.latvec.e21 = ucell.latvec.e23 = ucell.latvec.e31 = ucell.latvec.e32 = 0.0;

    ucell.atoms[0].taud[0].set(0.0, 0.0, 0.0);
    ucell.atoms[1].taud[0].set(0.22, 0.22, 0.22);
    for (int it = 0; it < ucell.ntype; ++it)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            ucell.atoms[it].tau[ia] = ucell.atoms[it].taud[ia] * ucell.latvec;
            ucell.atoms[it].mbl[ia].set(1, 1, 1);
            ucell.atoms[it].vel[ia].set(0.0, 0.0, 0.0);
        }
    }

    ucell.omega = std::abs(ucell.latvec.Det()) * ucell.lat0 * ucell.lat0 * ucell.lat0;
    ucell.GT = ucell.latvec.Inverse();
    ucell.G = ucell.GT.Transpose();
    ucell.GGT = ucell.G * ucell.GT;
    ucell.invGGT = ucell.GGT.Inverse();
    ucell.GT0 = ucell.GT;
    ucell.G0 = ucell.G;
    ucell.GGT0 = ucell.GGT;
    ucell.invGGT0 = ucell.invGGT;

    ucell.set_iat2itia();
}

bool finite_vec(const ModuleBase::Vector3<double>* f, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            if (!std::isfinite(f[i][k]))
            {
                return false;
            }
        }
    }
    return true;
}
} // namespace

#ifdef __DPMD

class MD_dp_mdfunc_Test : public ::testing::Test
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
        PARAM.sys.global_out_dir = "./";
        PARAM.sys.global_readin_dir = "./";
        GlobalV::ofs_running.open("/dev/null");

        setup_minimal_dp_ucell(ucell, inp);
        force = new ModuleBase::Vector3<double>[ucell.nat];
        virial.create(3, 3);
        stress.create(3, 3);
    }

    void TearDown() override
    {
        delete[] force;
        delete[] ucell.atoms;
        GlobalV::ofs_running.close();
    }
};

TEST_F(MD_dp_mdfunc_Test, force_virial_deepmd_consistent)
{
    ModuleESolver::ESolver* p_esolver = new ModuleESolver::ESolver_DP("./case_1.pb");
    p_esolver->before_all_runners(ucell, inp);

    double pe1 = 0.0, pe2 = 0.0;
    MD_func::force_virial(p_esolver, 0, ucell, pe1, force, true, virial);

    std::vector<ModuleBase::Vector3<double>> vel0(ucell.nat);
    std::vector<double> mass(ucell.nat, 1.0);
    MD_func::compute_stress(ucell, vel0.data(), mass.data(), true, virial, stress);

    EXPECT_TRUE(std::isfinite(pe1));
    EXPECT_TRUE(finite_vec(force, ucell.nat));
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_TRUE(std::isfinite(virial(i, j)));
        }
    }

    MD_func::force_virial(p_esolver, 1, ucell, pe2, force, true, virial);
    EXPECT_NEAR(pe1, pe2, 1.0e-12);

    delete p_esolver;
}

#else

TEST(MD_dp_mdfunc_placeholder, skipped_without_deepmd)
{
    GTEST_SKIP() << "Build with -DDeePMD_DIR=... to enable DeePMD MD_func tests.";
}

#endif
