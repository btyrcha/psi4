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

#ifndef FDDS_DISP_H
#define FDDS_DISP_H

#include "psi4/libmints/typedefs.h"

namespace psi {

class BasisSet;
class DFHelper;

namespace sapt {

class FDDS_Dispersion {
   protected:
    // BasisSets
    std::shared_ptr<BasisSet> primary_;
    std::shared_ptr<BasisSet> auxiliary_;

    // Coulomb metric inverse
    SharedMatrix metric_inv_;

    // Coulomb metric
    SharedMatrix metric_;

    // Coulomb metric -0.5 power
    SharedMatrix metric_half_inv_;

    // Auxiliary overlap matrix
    SharedMatrix aux_overlap_;

    // DFHelper object
    std::shared_ptr<DFHelper> dfh_;

    // Cache map
    std::map<std::string, SharedMatrix> matrix_cache_;
    std::map<std::string, SharedVector> vector_cache_;

    // Is hybrid functional? 
    bool is_hybrid_;

    // QR factorization result
    SharedMatrix R_A_, R_B_;

    /**
     * Writes M, nocc * nvir rows in (ar) order and n columns, to disk as the tensors ar_name (nocc, nvir, n)
     * and ra_name (nvir, nocc, n)
     * @param tmp Scratch matrix of the same shape as M
     */
    void write_ar_ra(SharedMatrix M, SharedMatrix tmp, size_t nocc, size_t nvir, const std::string& ar_name,
                     const std::string& ra_name);

   public:
    /// One contraction in form_X/form_Y: input and output disk tensor names and their last dimension
    struct XYContraction {
        std::string in;
        std::string out;
        size_t ncol;
    };

    /**
     * Constructs the FDDS_Dispersion object.
     * @param primary   The primary basis
     * @param auxiliary The auxiliary basis
     * @param cache     A data cache containing "Cocc_A", "Cvir_A", "eps_occ_A", "eps_vir_A", "Cocc_B",
     * "Cvir_B", "eps_occ_B", "eps_vir_B" quantities
     * @param is_hybrid Flag of hybrid functional
     */
    FDDS_Dispersion(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                    std::map<std::string, SharedMatrix> matrix_cache, std::map<std::string, SharedVector> vector_cache,
                    bool is_hybrid);

    ~FDDS_Dispersion();

    /**
     * Projects the densities from primary AO space to auxiliary AO space
     * @param  dens     Vector of densities to transform
     * @return aux_dens Vector of transformed densities
     */
    std::vector<SharedMatrix> project_densities(std::vector<SharedMatrix> dens);

    /**
     * Forms the uncoupled amplitude Qia,Eia,Pia->PQ and projector Qia,Pia->PQ
     * @param  monomer Monomer "A" or "B"
     * @param  omega   Time dependent value
     * @return         "PQ" amplitude 
     */
    SharedMatrix form_unc_amplitude(std::string monomer, double omega);

    /**
     * Forms the uncoupled amplitude and other PQ matrices in hybrid FDDS dispersion
     * @param  monomer Monomer "A" or "B"
     * @param  omega   Time dependent value
     * @return         Dictionary of PQ matrics 
     * @return ret["amp"]  PQ uncoupled amplitude
     * @return ret["K1LD"]  K1(lambda * d) = (P|ar) LDar [(X+Y) Q]_ar
     * @return ret["K2LD"]  K2(lambda * d) = (P|ar) Lar [(Y-X) d Q]_ar
     * @return ret["K2L"]  K2(lambda) = (P|ar) Lar [(Y-X) (ar|Q)]_ar
     * @return ret["K21L"]  K21(lambda) = (P|ar) Lar [(Y-X) (X+Y) Q]_ar
     * with X[ar,a'r'] = (ar'|a'r), Y[ar,a'r'] = (aa'|rr'), d = e_r - e_a, L = -4 / (d^2 + omega^2) and Q of QR().
     * Lambda and d sit where the Woodbury expansion of Eq. (39) of Xie et al., JCP 157, 024801 (2022) puts
     * them; Eqs. (45), (46), (57) and (58) there place them differently.
     * amp and K2L are naux x naux; K1LD, K2LD and K21L are naux x k, with k = min(nocc * nvir, naux)
     */
    std::map<std::string, SharedMatrix> form_aux_matrices(std::string monomer, double omega);

    /**
     * Returns the metric matrix
     * @return Metric
     */
    SharedMatrix metric() { return metric_; }

    /**
     * Returns the metric_inv matrix
     * @return Metric
     */
    SharedMatrix metric_inv() { return metric_inv_; }

    /**
     * Returns the auxiliary overlap matrix
     * @return Overlap
     */
    SharedMatrix aux_overlap() { return aux_overlap_; }

    /**
     * Returns R for the economy QR decomposition of (ar|Q), k x naux with k = min(nocc * nvir, naux)
     * @return R_A
     */
    SharedMatrix R_A() { return R_A_; }

    /**
     * Returns R for the economy QR decomposition of (bs|Q), k x naux with k = min(nocc * nvir, naux)
     * @return R_B
     */
    SharedMatrix R_B() { return R_B_; }

    /**
     * Contracts the X-type exchange matrix X[ar,a'r'] = (ar'|a'r) with 3-index tensors,
     * (ar|out|Q) = sum_a'r' (ar'|a'r) (a'r'|in|Q), all in one pass over the integrals
     * @param monomer Monomer "A" or "B"
     * @param in_out  in: existing (nocc, nvir, ncol) disk tensors; out: created with the same shape
     */
    void form_X(std::string monomer, const std::vector<XYContraction>& in_out);

    /**
     * Contracts the Y-type exchange matrix Y[ar,a'r'] = (aa'|rr') with 3-index tensors,
     * (ar|out|Q) = sum_a'r' (aa'|rr') (r'a'|in|Q), all in one pass over the integrals
     * @param monomer Monomer "A" or "B"
     * @param in_out  in: existing (nvir, nocc, ncol) disk tensors, (ra) order; out: created as (nocc, nvir, ncol)
     */
    void form_Y(std::string monomer, const std::vector<XYContraction>& in_out);

    /**
     * Writes (X+Y) Q = (ar|QX|Q) + (ar|QY|Q) to disk in (ar) and (ra) order, as (ar|QP|Q) and (ra|QP|Q),
     * the input of the form_X/form_Y pass that gives (Y-X) (X+Y) Q for K21
     * @param monomer Monomer "A" or "B"
     */
    void form_XpYQ(std::string monomer);

    /**
     * Performs the economy QR factorization (ar|Q) = Q R, with k = min(nocc * nvir, naux) columns of Q.
     * Writes Q to disk as (ar|Q|Q) and (ra|Q|Q), and d Q (d = e_r - e_a) as (ar|QD|Q) and (ra|QD|Q),
     * all k wide in the last index.
     * @param monomer Monomer "A" or "B"
     * @return R, k x naux, stored into R_A_ or R_B_ by the caller
     */
    SharedMatrix QR(std::string monomer);

    SharedMatrix get_tensor_pqQ(std::string name, std::tuple<size_t, size_t, size_t> dimensions);
    void print_tensor_pqQ(std::string tensor_name, std::string file_name, std::tuple<size_t, size_t, size_t> dimensions);

};  // End FDDS_Dispersion
}  // namespace sapt
}  // namespace psi

#endif
