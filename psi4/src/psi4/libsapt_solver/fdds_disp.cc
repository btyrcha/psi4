/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "psi4/libqt/qt.h"
#include "psi4/psi4-dec.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libsapt_solver/fdds_disp.h"
#include "psi4/libmints/3coverlap.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/lib3index/dfhelper.h"

#include <iomanip>

// OMP
#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi {

namespace sapt {

FDDS_Dispersion::FDDS_Dispersion(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                                 std::map<std::string, SharedMatrix> matrix_cache,
                                 std::map<std::string, SharedVector> vector_cache, 
                                 bool is_hybrid)
    : primary_(primary), auxiliary_(auxiliary), matrix_cache_(matrix_cache), vector_cache_(vector_cache), is_hybrid_(is_hybrid) {

    // ==> Check incoming cache <==
    std::vector<std::string> matrix_cache_check = {"Cocc_A", "Cvir_A", "Cocc_B", "Cvir_B"};
    for (auto key : matrix_cache_check) {
        if (matrix_cache_.find(key) == matrix_cache_.end()) {
            outfile->Printf("FDDS_Dispersion: Missing matrix_cache key %s\n", key.c_str());
            throw PSIEXCEPTION("FDDS_Dispersion: Missing values in the matrix_cache!");
        }
    }

    std::vector<std::string> vector_cache_check = {"eps_occ_A", "eps_vir_A", "eps_occ_B", "eps_vir_B"};
    for (auto key : vector_cache_check) {
        if (vector_cache_.find(key) == vector_cache_.end()) {
            outfile->Printf("FDDS_Dispersion: Missing vector_cache key %s\n", key.c_str());
            throw PSIEXCEPTION("FDDS_Dispersion: Missing values in the vector_cache!");
        }
    }

    // ==> Form Metric <==
   
    timer_on("Form JS");
    size_t naux = auxiliary_->nbf();
    metric_ = std::make_shared<Matrix>("Inv Coulomb Metric", naux, naux);

    std::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    size_t nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif

    // ==> (P|Q) Metric <==
    IntegralFactory metric_factory(auxiliary_, zero, auxiliary_, zero);

    std::vector<std::shared_ptr<TwoBodyAOInt>> metric_ints(nthread);
    std::vector<const double*> metric_buff(nthread);
    for (size_t thread = 0; thread < nthread; thread++) {
        metric_ints[thread] = std::shared_ptr<TwoBodyAOInt>(metric_factory.eri());
        metric_buff[thread] = metric_ints[thread]->buffer();
    }

    double** metricp = metric_->pointer();

#pragma omp parallel for schedule(dynamic) num_threads(nthread)
    for (size_t MU = 0; MU < auxiliary_->nshell(); ++MU) {
        size_t nummu = auxiliary_->shell(MU).nfunction();

        size_t thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        // Triangular
        for (size_t NU = 0; NU <= MU; ++NU) {
            size_t numnu = auxiliary_->shell(NU).nfunction();

            metric_ints[thread]->compute_shell(MU, 0, NU, 0);
            metric_buff[thread] = metric_ints[thread]->buffer();

            size_t index = 0;
            // #pragma simd collapse(2)
            for (size_t mu = 0; mu < nummu; ++mu) {
                size_t omu = auxiliary_->shell(MU).function_index() + mu;

                for (size_t nu = 0; nu < numnu; ++nu, ++index) {
                    size_t onu = auxiliary_->shell(NU).function_index() + nu;

                    metricp[omu][onu] = metricp[onu][omu] = metric_buff[thread][index];
                }
            }
        }
    }

    metric_ints.clear();
    metric_buff.clear();

    metric_inv_ = metric_->clone();
    metric_inv_->power(-1.0, 1.e-12);

    // ==> Form Aux overlap <==

    IntegralFactory factory(auxiliary_);
    std::shared_ptr<OneBodyAOInt> overlap(factory.ao_overlap());
    aux_overlap_ = std::make_shared<Matrix>("Auxiliary Overlap", naux, naux);
    overlap->compute(aux_overlap_);
    timer_off("Form JS");

    // ==> Form 3-index object <==

    // Build C Stack
    std::vector<SharedMatrix> Cstack_vec;
    Cstack_vec.push_back(matrix_cache_["Cocc_A"]);
    Cstack_vec.push_back(matrix_cache_["Cvir_A"]);
    Cstack_vec.push_back(matrix_cache_["Cocc_B"]);
    Cstack_vec.push_back(matrix_cache_["Cvir_B"]);

    size_t doubles = Process::environment.get_memory() * 0.8 / sizeof(double);
    size_t max_MO = 0;
    for (auto& mat : Cstack_vec) max_MO = std::max(max_MO, (size_t)mat->ncol());

    // Build DFHelper
    dfh_ = std::make_shared<DFHelper>(primary_, auxiliary_);
    dfh_->set_memory(doubles);
    if (is_hybrid_) {
        dfh_->set_method("DIRECT");
    } else {
        dfh_->set_method("DIRECT_iaQ");
    }
    dfh_->set_nthreads(nthread);
    dfh_->set_metric_pow(0.0);
    dfh_->initialize();
    dfh_->print_header();

    // Define spaces
    dfh_->add_space("a", Cstack_vec[0]);
    dfh_->add_space("r", Cstack_vec[1]);
    dfh_->add_space("b", Cstack_vec[2]);
    dfh_->add_space("s", Cstack_vec[3]);

    // add transformations
    dfh_->add_transformation("arQ", "a", "r", "pqQ");
    dfh_->add_transformation("bsQ", "b", "s", "pqQ");
    if (is_hybrid_) {
        dfh_->add_transformation("raQ", "r", "a", "pqQ");
        dfh_->add_transformation("sbQ", "s", "b", "pqQ");
        dfh_->add_transformation("Qar", "a", "r", "Qpq");
        dfh_->add_transformation("Qbs", "b", "s", "Qpq");
    }

    // transform
    dfh_->set_release_core_AO_before_metric(true);
    dfh_->transform();

    // transformations specific for hybrid functional

    if (is_hybrid_) {
        // Contracted 3-index integrals to reproduce 4-index ERI
        // Clear spaces to re-order spaces and transformations in DFHelper
        // Clear transformations to avoid overwriting pqQ tensors 
        dfh_->clear_spaces();
        dfh_->clear_transformations();
        dfh_->set_method("DIRECT_iaQ");
        dfh_->set_metric_pow(-0.5);
        dfh_->initialize();

        dfh_->add_space("a", Cstack_vec[0]);
        dfh_->add_space("r", Cstack_vec[1]);
        dfh_->add_space("b", Cstack_vec[2]);
        dfh_->add_space("s", Cstack_vec[3]);

        dfh_->add_transformation("aaR", "a", "a", "pqQ");
        dfh_->add_transformation("arR", "a", "r", "pqQ");
        dfh_->add_transformation("rrR", "r", "r", "pqQ");
        dfh_->add_transformation("bbR", "b", "b", "pqQ");
        dfh_->add_transformation("bsR", "b", "s", "pqQ");
        dfh_->add_transformation("ssR", "s", "s", "pqQ");
        dfh_->set_release_core_AO_before_metric(true);
        dfh_->transform();
    }

    dfh_->clear_spaces();

    if (is_hybrid_) {
        // QR Factorization of (ar|Q)
        timer_on("FDDS: QR");
        R_A_ = QR("A");
        R_B_ = QR("B");
        timer_off("FDDS: QR");

        // Tensor names: <pre>arQ and <pre>raQ for monomer A, <pre>bsQ and <pre>sbQ for monomer B
        auto ov = [](const std::string& mon, const std::string& pre) { return pre + (mon == "A" ? "ar" : "bs") + "Q"; };
        auto vo = [](const std::string& mon, const std::string& pre) { return pre + (mon == "A" ? "ra" : "sb") + "Q"; };
        size_t naux = auxiliary_->nbf();
        // Q-derived tensors are k wide, k = min(nocc * nvir, naux), as Q in QR()
        auto nk = [&](const std::string& mon) {
            size_t nov = (size_t)vector_cache_["eps_occ_" + mon]->dim(0) * vector_cache_["eps_vir_" + mon]->dim(0);
            return std::min(nov, naux);
        };

        // form (ar|X|Q), (ar|QX|Q) and (ar|QDX|Q): X applied to (ar|Q), Q and d Q
        timer_on("FDDS: Form X");
        for (std::string mon : {"A", "B"}) {
            form_X(mon, {{ov(mon, ""), ov(mon, "X"), naux},
                         {ov(mon, "Q"), ov(mon, "QX"), nk(mon)},
                         {ov(mon, "QD"), ov(mon, "QDX"), nk(mon)}});
        }
        timer_off("FDDS: Form X");

        // form (ar|Y|Q), (ar|QY|Q) and (ar|QDY|Q): Y applied to (ar|Q), Q and d Q
        timer_on("FDDS: Form Y");
        for (std::string mon : {"A", "B"}) {
            form_Y(mon, {{vo(mon, ""), ov(mon, "Y"), naux},
                         {vo(mon, "Q"), ov(mon, "QY"), nk(mon)},
                         {vo(mon, "QD"), ov(mon, "QDY"), nk(mon)}});
        }
        timer_off("FDDS: Form Y");

        // form (ar|QPX|Q) and (ar|QPY|Q): X and Y applied to (X+Y) Q, for K21
        timer_on("FDDS: Form X/Y (X+Y)Q");
        for (std::string mon : {"A", "B"}) {
            form_XpYQ(mon);
            form_X(mon, {{ov(mon, "QP"), ov(mon, "QPX"), nk(mon)}});
            form_Y(mon, {{vo(mon, "QP"), ov(mon, "QPY"), nk(mon)}});
        }
        timer_off("FDDS: Form X/Y (X+Y)Q");
    }

}

FDDS_Dispersion::~FDDS_Dispersion() {}

std::vector<SharedMatrix> FDDS_Dispersion::project_densities(std::vector<SharedMatrix> densities) {
    // Perform the contraction
    // (PQS) (S|R)^-1 (R|pq) Dpq -> PQ

    // ==> Contract (R|pq) Dpq -> R <== //

    std::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    size_t nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif

    // Build integral threads
    IntegralFactory df_factory(auxiliary_, zero, primary_, primary_);

    std::vector<std::shared_ptr<TwoBodyAOInt>> df_ints(nthread);
    std::vector<const double*> df_buff(nthread);
    for (size_t thread = 0; thread < nthread; thread++) {
        df_ints[thread] = std::shared_ptr<TwoBodyAOInt>(df_factory.eri());
        df_buff[thread] = df_ints[thread]->buffer();
    }

    size_t naux = auxiliary_->nbf();
    size_t nbf = primary_->nbf();
    size_t nbf2 = nbf * nbf;

    // Pack the DF pairs
    std::vector<std::pair<size_t, size_t>> df_pairs;
    for (size_t M = 0; M < primary_->nshell(); M++) {
        for (size_t N = 0; N <= M; N++) {
            df_pairs.push_back(std::pair<size_t, size_t>(M, N));
        }
    }

    // Check on memory real quick
    size_t doubles = Process::environment.get_memory() * 0.8 / sizeof(double);
    size_t mem_size = nbf2 * auxiliary_->max_nprimitive() * nthread;
    if (mem_size > doubles) {
        std::stringstream message;
        double mem_gb = ((double)(mem_size) / 0.8 * sizeof(double));
        message << "FDDS Dispersion requires at least nbf^2 * max_ang * nthread of memory." << std::endl;
        message << "       After taxes this is " << std::setprecision(2) << mem_gb << " GB of memory.";
        throw PSIEXCEPTION(message.str());
    }

    // Build result and temp vectors
    std::vector<SharedVector> aux_dens;
    for (size_t i = 0; i < densities.size(); i++) {
        aux_dens.push_back(std::make_shared<Vector>(naux));
    }

    std::vector<SharedMatrix> collapse_temp;
    for (size_t i = 0; i < nthread; i++) {
        collapse_temp.push_back(std::make_shared<Matrix>(auxiliary_->max_function_per_shell(), nbf2));
    }

// Do the contraction
#pragma omp parallel for schedule(dynamic) num_threads(nthread)
    for (size_t Rshell = 0; Rshell < auxiliary_->nshell(); Rshell++) {
        size_t thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        collapse_temp[thread]->zero();
        double** tempp = collapse_temp[thread]->pointer();

        size_t num_r = auxiliary_->shell(Rshell).nfunction();
        size_t index_r = auxiliary_->shell(Rshell).function_index();

        // Loop over our PQ shells
        for (auto PQshell : df_pairs) {
            size_t Pshell = PQshell.first;
            size_t Qshell = PQshell.second;

            df_ints[thread]->compute_shell(Rshell, 0, Pshell, Qshell);
            df_buff[thread] = df_ints[thread]->buffer();

            size_t num_p = primary_->shell(Pshell).nfunction();
            size_t index_p = primary_->shell(Pshell).function_index();

            size_t num_q = primary_->shell(Qshell).nfunction();
            size_t index_q = primary_->shell(Qshell).function_index();

            size_t index = 0;
            for (size_t r = 0; r < num_r; r++) {
                for (size_t p = index_p; p < index_p + num_p; p++) {
                    for (size_t q = index_q; q < index_q + num_q; q++) {
                        tempp[r][p * nbf + q] = tempp[r][q * nbf + p] = df_buff[thread][index++];
                    }
                }
            }
        }

        // Stitch it together
        for (size_t i = 0; i < densities.size(); i++) {
            C_DGEMV('N', num_r, nbf2, 1.0, tempp[0], nbf2, densities[i]->pointer()[0], 1, 0.0,
                    (aux_dens[i]->pointer() + index_r), 1);
        }

    }  // End Rshell

    // Clear a few temps
    collapse_temp.clear();
    df_buff.clear();
    df_ints.clear();
    df_pairs.clear();

    // ==> Contract (S|R)^-1 R -> S <== //
    std::vector<SharedVector> aux_dens_inv;
    for (size_t i = 0; i < densities.size(); i++) {
        aux_dens_inv.push_back(std::make_shared<Vector>(naux));
        aux_dens_inv[i]->gemv(false, 1.0, *metric_inv_, *aux_dens[i], 0.0);
    }

    // ==> Contract (PQS) S -> PQ <== //
    std::vector<std::shared_ptr<ThreeCenterOverlapInt>> aux_ints(nthread);
    std::vector<const double*> aux_buff(nthread);

    for (size_t i = 0; i < nthread; i++) {
        aux_ints[i] = std::shared_ptr<ThreeCenterOverlapInt>(
            new ThreeCenterOverlapInt(auxiliary_, auxiliary_, auxiliary_));
        aux_buff[i] = aux_ints[i]->buffers()[0];
    }

    // Pack the Aux pairs
    std::vector<std::pair<size_t, size_t>> aux_pairs;
    for (size_t M = 0; M < auxiliary_->nshell(); M++) {
        for (size_t N = 0; N <= M; N++) {
            aux_pairs.push_back(std::pair<size_t, size_t>(M, N));
        }
    }

    // Build result and temp vectors
    std::vector<SharedMatrix> ret;
    for (size_t i = 0; i < densities.size(); i++) {
        ret.push_back(std::make_shared<Matrix>(naux, naux));
    }

    size_t max_func = auxiliary_->max_function_per_shell();
    for (size_t i = 0; i < nthread; i++) {
        collapse_temp.push_back(std::make_shared<Matrix>(max_func * max_func, naux));
    }

#pragma omp parallel for schedule(dynamic) num_threads(nthread)
    for (size_t PQi = 0; PQi < aux_pairs.size(); PQi++) {
        size_t thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        size_t Pshell = aux_pairs[PQi].first;
        size_t Qshell = aux_pairs[PQi].second;

        size_t num_p = auxiliary_->shell(Pshell).nfunction();
        size_t index_p = auxiliary_->shell(Pshell).function_index();

        size_t num_q = auxiliary_->shell(Qshell).nfunction();
        size_t index_q = auxiliary_->shell(Qshell).function_index();

        double** tempp = collapse_temp[thread]->pointer();

        // Build aux temp
        for (size_t Rshell = 0; Rshell < auxiliary_->nshell(); Rshell++) {
            size_t num_r = auxiliary_->shell(Rshell).nfunction();
            size_t index_r = auxiliary_->shell(Rshell).function_index();

            aux_ints[thread]->compute_shell(Pshell, Qshell, Rshell);
            aux_buff[thread] = aux_ints[thread]->buffers()[0];

            size_t index = 0;
            for (size_t p = 0; p < num_p; p++) {
                for (size_t q = 0; q < num_q; q++) {
                    for (size_t r = index_r; r < num_r + index_r; r++) {
                        tempp[p * num_q + q][r] = aux_buff[thread][index++];
                    }
                }
            }
        }

        // Contract back to aux
        for (size_t i = 0; i < densities.size(); i++) {
            double** retp = ret[i]->pointer();
            double* dens = aux_dens_inv[i]->pointer();
            for (size_t p = 0; p < num_p; p++) {
                size_t abs_p = index_p + p;
                for (size_t q = 0; q < num_q; q++) {
                    size_t abs_q = index_q + q;

                    retp[abs_p][abs_q] = retp[abs_q][abs_p] = 2.0 * C_DDOT(naux, tempp[p * num_q + q], 1, dens, 1);
                }
            }
        }
    }

    return ret;
}

SharedMatrix FDDS_Dispersion::form_unc_amplitude(std::string monomer, double omega) {
    // ==> Configuration <==
    SharedVector eps_occ, eps_vir;
    std::string ovQ_tensor_name;

    if (monomer == "A") {
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
        ovQ_tensor_name = "arQ";
    } else if (monomer == "B") {
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
        ovQ_tensor_name = "bsQ";

    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::form_unc_amplitude: Monomer must be A or B!");
    }

    // Sizes
    size_t nocc = eps_occ->dim(0);
    size_t nvir = eps_vir->dim(0);
    size_t naux = auxiliary_->nbf();

    // Check on memory real quick
    size_t doubles = Process::environment.get_memory() * 0.8 / sizeof(double);
    size_t mem_size = 2 * naux * nvir + naux * naux + nvir * nocc;
    if (mem_size > doubles) {
        std::stringstream message;
        double mem_gb = ((double)(mem_size) / 0.8 * sizeof(double));
        message << "FDDS Dispersion requires at least naux * nvir + naux * naux of memory." << std::endl;
        message << "       After taxes this is " << std::setprecision(2) << mem_gb << " GB of memory.";
        throw PSIEXCEPTION(message.str());
    }

    // ==> Uncoupled Amplitudes <==
    auto amp = std::make_shared<Matrix>(nocc, nvir);

    double** ampp = amp->pointer();
    double* eoccp = eps_occ->pointer();
    double* evirp = eps_vir->pointer();

#pragma omp parallel for
    for (size_t i = 0; i < nocc; i++) {
        for (size_t a = 0; a < nvir; a++) {
            double val = -1.0 * (eoccp[i] - evirp[a]);
            double tmp = 4.0 * val / (val * val + omega * omega);
            // Lets see how stable this is, should be fine
            if (tmp < 1.e-14) {
                ampp[i][a] = 0.0;
            } else {
                ampp[i][a] = std::pow(tmp, 0.5);
            }
        }
    }

    // amp->print();

    // ==> Contract <==

    size_t dmem = doubles - naux * naux - nvir * nocc;
    size_t bsize = dmem / (naux * nvir);
    if (bsize > nocc) {
        bsize = nocc;
    }
    size_t nblocks = 1 + ((nocc - 1) / bsize);

    // printf("dmem:    %zu\n", dmem);
    // printf("bsize:   %zu\n", bsize);


    auto ret = std::make_shared<Matrix>("UNC Amplitude", naux, naux);
    auto tmp = std::make_shared<Matrix>("arQ tmp", bsize * nvir, naux);

    double** tmpp = tmp->pointer();

    ret->zero();
    size_t rstat, osize;
    for (size_t block = 0, bcount = 0; block < nblocks; block++) {
        // printf("Block %zu\n", block);
        tmp->zero();
        if (((block + 1) * bsize) > nocc) {
            osize = nocc - block * bsize;
        } else {
            osize = bsize;
        }

        dfh_->fill_tensor(ovQ_tensor_name, tmp, {bcount, bcount + osize});
        size_t shift_i = block * bsize;

#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < osize; i++) {
            for (size_t a = 0; a < nvir; a++) {
                double val = ampp[i + shift_i][a];
#pragma omp simd
                for (size_t Q = 0; Q < naux; Q++) {
                    tmpp[i * nvir + a][Q] *= val;
                }
            }
        }

        ret->gemm(true, false, 1.0, tmp, tmp, 1.0);
        bcount += osize;
    }

    return ret;
}

std::map<std::string, SharedMatrix> FDDS_Dispersion::form_aux_matrices(std::string monomer, double omega){

    // => Configuration <= //
    SharedVector eps_occ, eps_vir;
    std::string arQ_name, XarQ_name, YarQ_name, QParQ_name, QDXarQ_name, QDYarQ_name, QPXarQ_name, QPYarQ_name;

    if (monomer == "A") {
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
        arQ_name = "arQ";
        XarQ_name = "XarQ";
        YarQ_name = "YarQ";
        QParQ_name = "QParQ";
        QDXarQ_name = "QDXarQ";
        QDYarQ_name = "QDYarQ";
        QPXarQ_name = "QPXarQ";
        QPYarQ_name = "QPYarQ";
    } else if (monomer == "B") {
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
        arQ_name = "bsQ";
        XarQ_name = "XbsQ";
        YarQ_name = "YbsQ";
        QParQ_name = "QPbsQ";
        QDXarQ_name = "QDXbsQ";
        QDYarQ_name = "QDYbsQ";
        QPXarQ_name = "QPXbsQ";
        QPYarQ_name = "QPYbsQ";
    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::form_aux_matrices: Monomer must be A or B!");
    }


    // => Sizing <= //

    size_t nocc = eps_occ->dim(0);
    size_t nvir = eps_vir->dim(0);
    size_t naux = auxiliary_->nbf();
    size_t nk = std::min(nocc * nvir, naux);  // columns of Q in QR(); Q-derived tensors are nk wide

    // => Blocking <= //

    size_t doubles = Process::environment.get_memory() * 0.8 / sizeof(double);
    long long int rem = doubles - 2 * nocc * nvir - 6 * naux * naux;
    if (rem < 0)
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_aux_matrices()");

    // five slices naux wide and five nk wide
    size_t maxo = rem / (5 * nvir * (naux + nk));
    maxo = (maxo > nocc ? nocc : maxo);
    if (maxo < 1)
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_aux_matrices()");

    // => Scalars <= //

    auto Lar = std::make_shared<Matrix>(nocc, nvir); // lambda
    auto LDar = std::make_shared<Matrix>(nocc, nvir); // lambda * d

    double** Larp = Lar->pointer();
    double** LDarp = LDar->pointer();
    double* eoccp = eps_occ->pointer();
    double* evirp = eps_vir->pointer();

#pragma omp parallel for
    for (size_t a = 0; a < nocc; a++) {
        for (size_t r = 0; r < nvir; r++) {
            double val = evirp[r] - eoccp[a]; // d
            double ll = -4.0 / (val * val + omega * omega); // lambda
            double ld = val * ll; // lambda * d
            Larp[a][r] = ll;
            LDarp[a][r] = ld;
        }
    }

    // => Tensor Slices <= //

    auto arQ = std::make_shared<Matrix>("arQ", maxo * nvir, naux);
    auto arQLD = std::make_shared<Matrix>("arQLD", maxo * nvir, naux);
    auto XarQ = std::make_shared<Matrix>("XarQ", maxo * nvir, naux);
    auto YarQ = std::make_shared<Matrix>("YarQ", maxo * nvir, naux);
    auto YarQL = std::make_shared<Matrix>("YarQL", maxo * nvir, naux);
    auto QParQ = std::make_shared<Matrix>("QParQ", maxo * nvir, nk);    // (X+Y) Q
    auto QDXarQ = std::make_shared<Matrix>("QDXarQ", maxo * nvir, nk);  // X d Q
    auto QDYarQ = std::make_shared<Matrix>("QDYarQ", maxo * nvir, nk);  // Y d Q
    auto QPXarQ = std::make_shared<Matrix>("QPXarQ", maxo * nvir, nk);  // X (X+Y) Q
    auto QPYarQ = std::make_shared<Matrix>("QPYarQ", maxo * nvir, nk);  // Y (X+Y) Q

    // => Target <= //
    std::map<std::string, SharedMatrix> ret;
    ret["amp"] = std::make_shared<Matrix>("amp", naux, naux); // Uncoupled amplitude
    ret["K1LD"] = std::make_shared<Matrix>("K1LD", naux, nk);
    ret["K2LD"] = std::make_shared<Matrix>("K2LD", naux, nk);
    ret["K2L"] = std::make_shared<Matrix>("K2L", naux, naux);
    ret["K21L"] = std::make_shared<Matrix>("K21L", naux, nk);

    // Zero out matrices
    
    for (auto const &mat : ret)
        mat.second->zero();

    // => Pointers <= //

    double** arQp = arQ->pointer();
    double** arQLDp = arQLD->pointer();
    double** YarQLp = YarQL->pointer();
    double** QParQp = QParQ->pointer();
    double** QDYarQp = QDYarQ->pointer();
    double** QPYarQp = QPYarQ->pointer();

    double** ampp = ret["amp"]->pointer();
    double** K1LDp = ret["K1LD"]->pointer();
    double** K2LDp = ret["K2LD"]->pointer();
    double** K2Lp = ret["K2L"]->pointer();
    double** K21Lp = ret["K21L"]->pointer();

    // => Master Loop <= //

    for (size_t astart = 0; astart < nocc; astart += maxo) {
        size_t nablock = (astart + maxo >= nocc ? nocc - astart : maxo);

        size_t navir = nablock * nvir;

        dfh_->fill_tensor(arQ_name, arQ, {astart, astart + nablock});
        dfh_->fill_tensor(XarQ_name, XarQ, {astart, astart + nablock});
        dfh_->fill_tensor(YarQ_name, YarQ, {astart, astart + nablock});
        dfh_->fill_tensor(QParQ_name, QParQ, {astart, astart + nablock});
        dfh_->fill_tensor(QDXarQ_name, QDXarQ, {astart, astart + nablock});
        dfh_->fill_tensor(QDYarQ_name, QDYarQ, {astart, astart + nablock});
        dfh_->fill_tensor(QPXarQ_name, QPXarQ, {astart, astart + nablock});
        dfh_->fill_tensor(QPYarQ_name, QPYarQ, {astart, astart + nablock});

        // (Y-X) (ar|Q), (Y-X) d Q and (Y-X) (X+Y) Q
        YarQ->axpy(-1.0, XarQ);
        QDYarQ->axpy(-1.0, QDXarQ);
        QPYarQ->axpy(-1.0, QPXarQ);

        arQLD->copy(arQ);
        YarQL->copy(YarQ);

        // lambda goes to the left of (Y-X), as in Eq. (39)
#pragma omp parallel for collapse(2)
        for (size_t a = 0; a < nablock; a++) {
            for (size_t r = 0; r < nvir; r++) {
                double ll = Larp[astart + a][r];
                double ld = LDarp[astart + a][r];
#pragma omp simd
                for (size_t Q = 0; Q < naux; Q++) {
                    arQLDp[a * nvir + r][Q] *= ld;
                    YarQLp[a * nvir + r][Q] *= ll;
                }
#pragma omp simd
                for (size_t k = 0; k < nk; k++) {
                    QDYarQp[a * nvir + r][k] *= ll;
                    QPYarQp[a * nvir + r][k] *= ll;
                }
            }
        }

        C_DGEMM('T', 'N', naux, naux, navir, 1.0, arQLDp[0], naux, arQp[0], naux, 1.0, ampp[0], naux);
        C_DGEMM('T', 'N', naux, nk, navir, 1.0, arQLDp[0], naux, QParQp[0], nk, 1.0, K1LDp[0], nk);
        C_DGEMM('T', 'N', naux, nk, navir, 1.0, arQp[0], naux, QDYarQp[0], nk, 1.0, K2LDp[0], nk);
        C_DGEMM('T', 'N', naux, naux, navir, 1.0, arQp[0], naux, YarQLp[0], naux, 1.0, K2Lp[0], naux);
        C_DGEMM('T', 'N', naux, nk, navir, 1.0, arQp[0], naux, QPYarQp[0], nk, 1.0, K21Lp[0], nk);

    }
    
    return ret;
}

void FDDS_Dispersion::form_X(std::string monomer, const std::vector<XYContraction>& in_out) {

    // => Configuration <= //

    std::string arR_name;
    SharedVector eps_occ, eps_vir;
    
    if (monomer == "A") {
        arR_name = "arR";
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
    } else if (monomer == "B") {
        arR_name = "bsR";
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::form_X: Monomer must be A or B!");
    }


    // => Sizing <= //

    size_t nocc = eps_occ->dim(0);
    size_t nvir = eps_vir->dim(0);
    size_t naux = auxiliary_->nbf();
    size_t ncol_sum = 0;  // total width of the input (and of the output) tensors
    for (const auto& io : in_out) ncol_sum += io.ncol;

    // => Blocking <= //

    int nthread = 1;
#ifdef _OPENMP
    nthread = Process::environment.get_n_threads();
#endif

    size_t doubles = Process::environment.get_memory() * 0.8 / sizeof(double);
    long long int rem = doubles - nthread * nvir * nvir;
    if (rem < 0) 
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_X()");

    // two integral slices naux wide, plus an input and an output slice per contraction
    size_t maxo = rem / (2 * nvir * (naux + ncol_sum));
    maxo = (maxo > nocc ? nocc : maxo);
    if (maxo < 1) 
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_X()");

    // => Tensor Slices <= //

    auto arR = std::make_shared<Matrix>("arR", maxo * nvir, naux); // ([a]r'|R)
    auto aprR = std::make_shared<Matrix>("aprR", maxo * nvir, naux); // ([a']r|R)
    auto Vrr = std::make_shared<Matrix>("Vrr", nvir, nvir); // ([a']r|[a]r') single layer

    // ([a']r'|in|Q) and ([a]r|out|Q); each slice exactly as wide as its disk tensor
    std::vector<SharedMatrix> in_slices, out_slices;
    for (const auto& io : in_out) {
        in_slices.push_back(std::make_shared<Matrix>(io.in, maxo * nvir, io.ncol));
        out_slices.push_back(std::make_shared<Matrix>(io.out, maxo * nvir, io.ncol));
        dfh_->add_disk_tensor(io.out, std::make_tuple(nocc, nvir, io.ncol));  // (ar|out|Q) to disk
    }

    // => Pointers <= //

    double** arRp = arR->pointer();
    double** aprRp = aprR->pointer();
    double** Vrrp = Vrr->pointer();

    // => Master Loop <= //
    
    for (size_t astart = 0; astart < nocc; astart += maxo) {
        size_t nablock = (astart + maxo >= nocc ? nocc - astart : maxo);

        arR->zero();
        dfh_->fill_tensor(arR_name, arR, {astart, astart + nablock});
        for (auto& out : out_slices) out->zero();

        for (size_t apstart = 0; apstart < nocc; apstart += maxo) {
            size_t napblock = (apstart + maxo >= nocc ? nocc - apstart : maxo);        

            aprR->zero();   
            dfh_->fill_tensor(arR_name, aprR, {apstart, apstart + napblock});
            for (size_t i = 0; i < in_out.size(); i++) {
                in_slices[i]->zero();
                dfh_->fill_tensor(in_out[i].in, in_slices[i], {apstart, apstart + napblock});
            }

            size_t naap = nablock * napblock;

            for (size_t aap = 0; aap < naap; aap++) {
                size_t a = aap / napblock;
                size_t ap = aap % napblock;

                // => Contraction ((a')r|R) (R|(a)r') ((a')r'|Q) -> ((a)r|X|Q) for a, a' <= // This one is not correct. a (a') and r (r') have to be in the same bra/ket, because (ar|a'r') != (ar'|a'r). (ar|a'r') = (ar|R) (R|a'r') defines the ordering of 3-index integrals.
                // => Contraction [((a')r'|R) (R|(a)r)]^T ((a')r'|Q) -> ((a)r|X|Q) for a, a' <= // Correct expression in terms of (a'r'|R) and (ar|R).  
                // => Similar for every ((a)r|out|Q) <= //

                C_DGEMM('N', 'T', nvir, nvir, naux, 1.0, aprRp[ap * nvir], naux, arRp[a * nvir], naux, 0.0, Vrrp[0], nvir);
                for (size_t i = 0; i < in_out.size(); i++) {
                    size_t ncol = in_out[i].ncol;
                    C_DGEMM('N', 'N', nvir, ncol, nvir, 1.0, Vrrp[0], nvir, in_slices[i]->pointer()[ap * nvir], ncol,
                            1.0, out_slices[i]->pointer()[a * nvir], ncol);
                }
            }
        }

        for (size_t i = 0; i < in_out.size(); i++)
            dfh_->write_disk_tensor(in_out[i].out, out_slices[i], {astart, astart + nablock});
    }
}

void FDDS_Dispersion::form_Y(std::string monomer, const std::vector<XYContraction>& in_out) {

    // => Configuration <= //

    std::string aaR_name, rrR_name;
    SharedVector eps_occ, eps_vir;
    
    if (monomer == "A") {
        aaR_name = "aaR";
        rrR_name = "rrR";
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
    } else if (monomer == "B") {
        aaR_name = "bbR";
        rrR_name = "ssR";
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::form_Y: Monomer must be A or B!");
    }


    // => Sizing <= //

    size_t nocc = eps_occ->dim(0);
    size_t nvir = eps_vir->dim(0);
    size_t naux = auxiliary_->nbf();
    size_t ncol_sum = 0;  // total width of the input (and of the output) tensors
    for (const auto& io : in_out) ncol_sum += io.ncol;

    // => Blocking <= //

    int nthread = 1;
#ifdef _OPENMP
    nthread = Process::environment.get_n_threads();
#endif

    size_t doubles = Process::environment.get_memory() * 0.8 / sizeof(double);
    long long int rem = doubles - nthread * nocc * nvir;
    if (rem < 0) 
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_Y()");

    size_t kov = nvir / nocc + 1; // Ratio of v/o, take ceiling for worst case
    // aaR (maxo * nocc rows) and rrR (maxv * nvir rows) naux wide, plus per contraction an input slice
    // (maxv * nocc rows) and an output slice (maxo * nvir rows); with maxv = maxo * (kov - 1) and nvir < kov * nocc
    size_t maxo = rem / (((kov * kov - kov + 1) * naux + (2 * kov - 1) * ncol_sum) * nocc);
    size_t maxv = maxo * (kov - 1); 
    maxo = (maxo > nocc ? nocc : maxo);
    maxv = (maxv > nvir ? nvir : maxv);
    if (maxo < 1) 
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_Y()");

    // => Tensor Slices <= //

    auto aaR = std::make_shared<Matrix>("aaR", maxo * nocc, naux); // ([a]a'|R)
    auto rrR = std::make_shared<Matrix>("rrR", maxv * nvir, naux); // ([r']r|R)
    auto Vra = std::make_shared<Matrix>("Vra", nvir, nocc); // ([r']r|[a]a') single layer

    // ([r']a'|in|Q) and ([a]r|out|Q); each slice exactly as wide as its disk tensor
    std::vector<SharedMatrix> in_slices, out_slices;
    for (const auto& io : in_out) {
        in_slices.push_back(std::make_shared<Matrix>(io.in, maxv * nocc, io.ncol));
        out_slices.push_back(std::make_shared<Matrix>(io.out, maxo * nvir, io.ncol));
        dfh_->add_disk_tensor(io.out, std::make_tuple(nocc, nvir, io.ncol));  // (ar|out|Q) to disk
    }

    // => Pointers <= //

    double** aaRp = aaR->pointer();
    double** rrRp = rrR->pointer();
    double** Vrap = Vra->pointer();

    // => Master Loop <= //
    
    for (size_t astart = 0; astart < nocc; astart += maxo) {
        size_t nablock = (astart + maxo >= nocc ? nocc - astart : maxo);
        
        aaR->zero();
        dfh_->fill_tensor(aaR_name, aaR, {astart, astart + nablock});
        for (auto& out : out_slices) out->zero();

        for (size_t rstart = 0; rstart < nvir; rstart += maxv) {
            size_t nrblock = (rstart + maxv >= nvir ? nvir - rstart : maxv);        

            rrR->zero();
            dfh_->fill_tensor(rrR_name, rrR, {rstart, rstart + nrblock});
            for (size_t i = 0; i < in_out.size(); i++) {
                in_slices[i]->zero();
                dfh_->fill_tensor(in_out[i].in, in_slices[i], {rstart, rstart + nrblock});
            }

            size_t nar = nablock * nrblock;

            for (size_t ar = 0; ar < nar; ar++) {
                size_t a = ar / nrblock;
                size_t r = ar % nrblock;

                // => Contraction ((r')r|R) (R|(a)a') ((r')a'|Q) -> ((a)r|Y|Q) for a, r' <= //
                // Similar for every ((a)r|out|Q)

                C_DGEMM('N', 'T', nvir, nocc, naux, 1.0, rrRp[r * nvir], naux, aaRp[a * nocc], naux, 0.0, Vrap[0], nocc);
                for (size_t i = 0; i < in_out.size(); i++) {
                    size_t ncol = in_out[i].ncol;
                    C_DGEMM('N', 'N', nvir, ncol, nocc, 1.0, Vrap[0], nocc, in_slices[i]->pointer()[r * nocc], ncol,
                            1.0, out_slices[i]->pointer()[a * nvir], ncol);
                }
            }
        }

        for (size_t i = 0; i < in_out.size(); i++)
            dfh_->write_disk_tensor(in_out[i].out, out_slices[i], {astart, astart + nablock});
    }
}

void FDDS_Dispersion::form_XpYQ(std::string monomer) {

    // => Configuration <= //

    std::string QXarQ_name, QYarQ_name, QParQ_name, QPraQ_name;
    SharedVector eps_occ, eps_vir;

    if (monomer == "A") {
        QXarQ_name = "QXarQ";
        QYarQ_name = "QYarQ";
        QParQ_name = "QParQ";
        QPraQ_name = "QPraQ";
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
    } else if (monomer == "B") {
        QXarQ_name = "QXbsQ";
        QYarQ_name = "QYbsQ";
        QParQ_name = "QPbsQ";
        QPraQ_name = "QPsbQ";
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::form_XpYQ: Monomer must be A or B!");
    }

    // => Sizing <= //

    size_t nocc = eps_occ->dim(0);
    size_t nvir = eps_vir->dim(0);
    size_t naux = auxiliary_->nbf();
    size_t nk = std::min(nocc * nvir, naux);  // columns of Q in QR(); Q-derived tensors are nk wide

    // => Memory Check <= //

    size_t doubles = Process::environment.get_memory() * 0.8 / sizeof(double);
    if (doubles < 2 * nocc * nvir * nk) throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_XpYQ()");

    // (X+Y) Q, in core
    auto QP = std::make_shared<Matrix>("QP", nocc * nvir, nk);
    auto tmp = std::make_shared<Matrix>("tmp", nocc * nvir, nk);
    dfh_->fill_tensor(QXarQ_name, QP, {0, nocc});
    dfh_->fill_tensor(QYarQ_name, tmp, {0, nocc});
    QP->axpy(1.0, tmp);

    write_ar_ra(QP, tmp, nocc, nvir, QParQ_name, QPraQ_name);
}

void FDDS_Dispersion::write_ar_ra(SharedMatrix M, SharedMatrix tmp, size_t nocc, size_t nvir,
                                  const std::string& ar_name, const std::string& ra_name) {
    size_t ncol = M->colspi(0);

    dfh_->add_disk_tensor(ar_name, std::make_tuple(nocc, nvir, ncol));
    dfh_->write_disk_tensor(ar_name, M, {0, nocc});

    double** Mp = M->pointer();
    double** tmpp = tmp->pointer();
#pragma omp parallel for collapse(2)
    for (size_t a = 0; a < nocc; a++)
        for (size_t r = 0; r < nvir; r++) C_DCOPY(ncol, Mp[a * nvir + r], 1, tmpp[r * nocc + a], 1);

    dfh_->add_disk_tensor(ra_name, std::make_tuple(nvir, nocc, ncol));
    dfh_->write_disk_tensor(ra_name, tmp, {0, nvir});
}

SharedMatrix FDDS_Dispersion::QR(std::string monomer) {

    // => Configuration <= //

    std::string Qar_name, QarQ_name, QraQ_name, QDarQ_name, QDraQ_name;
    SharedVector eps_occ, eps_vir;
    
    if (monomer == "A") {
        Qar_name = "Qar";
        QarQ_name = "QarQ";
        QraQ_name = "QraQ";
        QDarQ_name = "QDarQ";
        QDraQ_name = "QDraQ";
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
    } else if (monomer == "B") {
        Qar_name = "Qbs";
        QarQ_name = "QbsQ";
        QraQ_name = "QsbQ";
        QDarQ_name = "QDbsQ";
        QDraQ_name = "QDsbQ";
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::QR: Monomer must be A or B!");
    }

    // => Sizing <= //

    size_t nocc = eps_occ->dim(0);
    size_t nvir = eps_vir->dim(0);
    size_t nbf = primary_->nbf();
    size_t naux = auxiliary_->nbf();    
    size_t nov = nocc * nvir;
    // Economy QR: (ar|Q) has rank <= min(nov, naux), so Q has nk orthonormal columns and R is nk x naux
    size_t nk = std::min(nov, naux);

    // => Meomry Check <= //

    size_t doubles = Process::environment.get_memory() * 0.8 / sizeof(double);
    size_t req_mem = nov * naux + nov * nk + nk * naux + nk;
    if (doubles < req_mem) throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::QR()");

    // => Tensor Slices <= //

    auto Qar = std::make_shared<Matrix>("Qar", naux, nov);
    dfh_->fill_tensor(Qar_name, Qar, {0, naux});

    // => Target <= //
    auto Q = std::make_shared<Matrix>("Q", nov, nk);
    auto tau = std::make_shared<Vector>("tau", nk);
    auto R = std::make_shared<Matrix>("R", nk, naux);

    // => Pointers <= //

    double** Qarp = Qar->pointer();
    double* taup = tau->pointer();
    double** Qp = Q->pointer();
    double** Rp = R->pointer();

    // => Work Buffer <= //

    double lwork_tmp, lwork_orgqr;
    C_DGEQRF(nov, naux, Qarp[0], nov, taup, &lwork_tmp, -1);
    C_DORGQR(nov, nk, nk, Qarp[0], nov, taup, &lwork_orgqr, -1);
    size_t lwork = (size_t)std::max(lwork_tmp, lwork_orgqr);
    auto work = std::make_shared<Vector>("work", lwork);
    double* workp = work->pointer();

    // Householder QR
    int info = C_DGEQRF(nov, naux, Qarp[0], nov, taup, workp, lwork);
    if (info != 0) {
        throw PSIEXCEPTION("FDDS_Dispersion::QR: DGEQRF failed with info = " + std::to_string(info));
    }

    // Save R, the upper trapezoid of the first nk rows
    R->zero();
    for (size_t row = 0; row < nk; row++)
        for (size_t col = row; col < naux; col++) {
            Rp[row][col] = Qarp[col][row];
        }

    // Transpose and Invert R
    // Only transpose, invert using np.linalg.pinv on numpy side; LAPACK subroutine DTRTRI is not stable
    // Return R as is

    // Form Q explicitly
    info = C_DORGQR(nov, nk, nk, Qarp[0], nov, taup, workp, lwork);
    if (info != 0) {
        throw PSIEXCEPTION("FDDS_Dispersion::QR: DORGQR failed with info = " + std::to_string(info));
    }
    size_t nar;
    // Q = Qar->transpose();
    for (size_t nQ = 0; nQ < nk; nQ++)
        for (size_t na = 0; na < nocc; na++)
            for (size_t nr = 0; nr < nvir; nr++) {
                nar = na * nvir + nr;
                Qp[nar][nQ] = Qarp[nQ][nar];
            }
    Qar.reset();
    work.reset();

    // Save Q as (ar|Q|Q) and (ra|Q|Q)
    auto tmp = std::make_shared<Matrix>("tmp", nov, nk);
    write_ar_ra(Q, tmp, nocc, nvir, QarQ_name, QraQ_name);

    // Save d Q as (ar|QD|Q) and (ra|QD|Q), d = e_r - e_a, for K2
    double* eoccp = eps_occ->pointer();
    double* evirp = eps_vir->pointer();
    for (size_t na = 0; na < nocc; na++)
        for (size_t nr = 0; nr < nvir; nr++) C_DSCAL(nk, evirp[nr] - eoccp[na], Qp[na * nvir + nr], 1);
    write_ar_ra(Q, tmp, nocc, nvir, QDarQ_name, QDraQ_name);

    return R;

}

SharedMatrix FDDS_Dispersion::get_tensor_pqQ(std::string name, std::tuple<size_t, size_t, size_t> dimensions) {

    // Debug helper
    // Returns as (pq|Q)
    size_t np = std::get<0>(dimensions);
    size_t nq = std::get<1>(dimensions);
    size_t nQ = std::get<2>(dimensions);

    auto M = std::make_shared<Matrix>(name, np * nq, nQ);

    dfh_->fill_tensor(name, M, {0, np});

    return M;
}


void FDDS_Dispersion::print_tensor_pqQ(std::string tensor_name, std::string file_name, std::tuple<size_t, size_t, size_t> dimensions) {

    // Debug helper
    size_t np = std::get<0>(dimensions);
    size_t nq = std::get<1>(dimensions);
    size_t nQ = std::get<2>(dimensions);

    auto M = std::make_shared<Matrix>(tensor_name, np * nq, nQ);
    dfh_->fill_tensor(tensor_name, M, {0, np});
    double** Mp = M->pointer();
    auto Mout = std::make_shared<PsiOutStream>(file_name);
    for (size_t row = 0; row < np * nq; row++)
        for (size_t col = 0; col < nQ; col++)
            Mout->Printf("%.15e\n", Mp[row][col]); 
}

}  // namespace sapt
}  // namespace psi
