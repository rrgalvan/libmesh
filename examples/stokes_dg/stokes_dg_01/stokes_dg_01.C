// The libMesh Finite Element Library.
// Copyright (C) 2002-2018 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


// <h1>RRGalvan Stokes DG 01 - Interior Penalty Discontinuous Galerkin
// for a System of Equations</h1> This example shows how a simple,
// linear system of equations, using DG, can be solved in parallel.
// The system of equations are two uncopuled Laplace equations.
//
// \author Rafa Rodríguez Galván \date 2018
//
// This example is based on Miscellaneous example 5 and on System of
// Equations example 1 (Stokes system).

#include <iostream>

// LibMesh include files.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/equation_systems.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_modification.h"
#include "libmesh/elem.h"
#include "libmesh/transient_system.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dense_submatrix.h"
#include "libmesh/dense_subvector.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/fe_interface.h"
#include "libmesh/getpot.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/error_vector.h"
#include "libmesh/kelly_error_estimator.h"
#include "libmesh/discontinuity_measure.h"
#include "libmesh/string_to_enum.h"

//#define QORDER TWENTYSIXTH

// Bring in everything from the libMesh namespace
using namespace libMesh;

void matrix_reposition( DenseSubMatrix<Number>& Kuu,
			DenseSubMatrix<Number>& Kuv,
			DenseSubMatrix<Number>& Kvu,
			DenseSubMatrix<Number>& Kvv,
			int u_var, int v_var,
			int n_rows_u, int n_cols_u, int n_rows_v, int n_cols_v )
{
  int u_first_row =  u_var*n_rows_u, v_first_row = v_var*n_rows_u;
  int u_first_col =  u_var*n_cols_u, v_first_col = v_var*n_cols_u;

  Kuu.reposition (u_first_row, u_first_col, n_rows_u, n_cols_u);
  Kuv.reposition (u_first_row, v_first_col, n_rows_u, n_cols_v);

  Kvu.reposition (v_first_row, u_first_col, n_rows_v, n_cols_u);
  Kvv.reposition (v_first_row, v_first_col, n_rows_v, n_cols_v);
}

// We now define the matrix assembly function for the
// Laplace system.  We need to first compute element volume
// matrices, and then take into account the boundary
// conditions and the flux integrals, which will be handled
// via an interior penalty method.
void assemble_stokesdg(EquationSystems & es,
                         const std::string & libmesh_dbg_var(system_name))
{
  libMesh::out << " assembling stokes dg system... ";
  libMesh::out.flush();

  // It is a good idea to make sure we are assembling
  // the proper system.
  libmesh_assert_equal_to (system_name, "StokesDG");

  // Get a constant reference to the mesh object.
  const MeshBase & mesh = es.get_mesh();
  // The dimension that we are running
  const unsigned int dim = mesh.mesh_dimension();

  // Get a reference to the LinearImplicitSystem we are solving
  LinearImplicitSystem & stokesdg_system = es.get_system<LinearImplicitSystem> ("StokesDG");
  // Numeric ids corresponding to each variable in the system
  const unsigned int u_var = stokesdg_system.variable_number ("u");
  const unsigned int v_var = stokesdg_system.variable_number ("v");

  // Get the Finite Element type for "u".  Note this will be
  // the same as the type for "v".
  FEType fe_u_type = stokesdg_system.variable_type(u_var);

  // Build a Finite Element object of the specified type.  Since the
  // FEBase::build() member dynamically creates memory we will
  // store the object as a std::unique_ptr<FEBase>.  This can be thought
  // of as a pointer that will clean up after itself.
  std::unique_ptr<FEBase> fe_u  (FEBase::build(dim, fe_u_type));
  std::unique_ptr<FEBase> fe_u_elem_face(FEBase::build(dim, fe_u_type));
  std::unique_ptr<FEBase> fe_u_neighbor_face(FEBase::build(dim, fe_u_type));

  // Get some parameters that we need during assembly
  const Real penalty = es.parameters.get<Real> ("penalty");
  std::string refinement_type = es.parameters.get<std::string> ("refinement");

  // A reference to the DofMap object for this system.  The DofMap
  // object handles the index translation from node and element numbers
  // to degree of freedom numbers.  We will talk more about the DofMap
  const DofMap & dof_map = stokesdg_system.get_dof_map();

  // Quadrature rules for numerical integration.
#ifdef QORDER
  QGauss qrule (dim, QORDER);
#else
  QGauss qrule (dim, fe_u_type.default_quadrature_order());
#endif
  fe_u->attach_quadrature_rule (&qrule);

#ifdef QORDER
  QGauss qface(dim-1, QORDER);
#else
  QGauss qface(dim-1, fe_u_type.default_quadrature_order());
#endif

  // Tell the finite element object to use our quadrature rule.
  fe_u_elem_face->attach_quadrature_rule(&qface);
  fe_u_neighbor_face->attach_quadrature_rule(&qface);

  // Here we define some references to cell-specific data that
  // will be used to assemble the linear system.
  // Data for interior volume integrals
  const std::vector<Real> & JxW = fe_u->get_JxW();
  const std::vector<std::vector<RealGradient>> & dphi = fe_u->get_dphi();

  // Data for surface integrals on the element boundary
  const std::vector<std::vector<Real>> &  phi_face = fe_u_elem_face->get_phi();
  const std::vector<std::vector<RealGradient>> & dphi_face = fe_u_elem_face->get_dphi();
  const std::vector<Real> & JxW_face = fe_u_elem_face->get_JxW();
  const std::vector<Point> & qface_normals = fe_u_elem_face->get_normals();
  const std::vector<Point> & qface_points = fe_u_elem_face->get_xyz();

  // Data for surface integrals on the neighbor boundary
  const std::vector<std::vector<Real>> &  phi_neighbor_face = fe_u_neighbor_face->get_phi();
  const std::vector<std::vector<RealGradient>> & dphi_neighbor_face = fe_u_neighbor_face->get_dphi();

  // Define data structures to contain the element interior matrix
  // and right-hand-side vector contribution.  Following
  // basic finite element terminology we will denote these
  // "Ke" and "Fe".
  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  DenseSubMatrix<Number>
    Ke_uu(Ke), Ke_uv(Ke),
    Ke_vu(Ke), Ke_vv(Ke);

  DenseSubVector<Number>
    Fe_u(Fe),
    Fe_v(Fe);

  // Data structures to contain the element and neighbor boundary
  // matrix contribution. For each face, this matrices will do the
  // coupling between the dofs of the element and those of his
  // neighbors.  Ken: matrix coupling elem and neighbor dofs
  DenseMatrix<Number> Kn;
  DenseMatrix<Number> Kne;
  DenseMatrix<Number> Ken;

  DenseSubMatrix<Number>
    Kn_uu(Kn), Kn_uv(Kn),
    Kn_vu(Kn), Kn_vv(Kn);
  DenseSubMatrix<Number>
    Kne_uu(Kne), Kne_uv(Kne),
    Kne_vu(Kne), Kne_vv(Kne);
  DenseSubMatrix<Number>
    Ken_uu(Ken), Ken_uv(Ken),
    Ken_vu(Ken), Ken_vv(Ken);

  // This vector will hold the degree of freedom indices for
  // the element.  These define where in the global system
  // the element degrees of freedom get mapped.
  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_u;
  std::vector<dof_id_type> dof_indices_v;
  std::vector<dof_id_type> dof_indices_p;

  // Now we will loop over all the elements in the mesh.  We will
  // compute first the element interior matrix and right-hand-side contribution
  // and then the element and neighbors boundary matrix contributions.
  for (const auto & elem : mesh.active_local_element_ptr_range())
    {
      // Get the degree of freedom indices for the
      // current element.  These define where in the global
      // matrix and right-hand-side this element will
      // contribute to.
      dof_map.dof_indices (elem, dof_indices);
      dof_map.dof_indices (elem, dof_indices_u, u_var);
      dof_map.dof_indices (elem, dof_indices_v, v_var);

      const unsigned int n_dofs = dof_indices.size();
      const unsigned int n_dofs_u = dof_indices_u.size();
      const unsigned int n_dofs_v = dof_indices_v.size();

      // Compute the element-specific data for the current
      // element.  This involves computing the location of the
      // quadrature points (q_point) and the shape functions
      // (phi, dphi) for the current element.
      fe_u->reinit (elem);

      // Zero the element matrix and right-hand side before
      // summing them.  We use the resize member here because
      // the number of degrees of freedom might have changed from
      // the last element.
      Ke.resize (n_dofs, n_dofs);
      Fe.resize (n_dofs);

      // Reposition the submatrices...  The idea is this:
      //
      //         -           -          -  -
      //        | Kuu Kuv Kup |        | Fu |
      //   Ke = | Kvu Kvv Kvp |;  Fe = | Fv |
      //        | Kpu Kpv Kpp |        | Fp |
      //         -           -          -  -
      //
      // Where:
      //
      //   * Kuu = sum_T int_T ( grad(u)*grad(ub) )
      //           + sum_e eta/h * int_e ( jump(u)*jump(ub) )  // Stability
      //           - sum_e int_e ( mean(grad(u))*n_e * jump(ub) )  // Consistency
      //           - sum_e int_e ( mean(grad(ub))*n_e * jump(u) )  // Consistency
      //   * Kuv = 0
      //   * Kup = - sum_T int_T ( p*dx(bu) )
      //           + sum_e int_e ( mean(p)*jump(ub)*(n_e)_x )
      //
      //   * Kvu = 0
      //   * Kvv = sum_T int_T ( grad(v)*grad(vb) )
      //           + sum_e eta/h * int_e ( jump(v)*jump(vb) )  // Stability
      //           - sum_e int_e ( mean(grad(v))*n_e * jump(vb) )  // Consistency
      //           - sum_e int_e ( mean(grad(vb))*n_e * jump(v) )  // Consistency
      //   * Kvp = - sum_T int_T ( p*dy(bv) )
      //           + sum_e int_e ( mean(p)*jump(vb)*(n_e)_y )
      //
      //   * Kpu = - sum_T int_T ( pb*dx(u) )
      //           + sum_e int_e ( mean(pb)*jump(u)*(n_e)_x )
      //   * Kpv = - sum_T int_T ( pb*dx(v) )
      //           + sum_e int_e ( mean(pb)*jump(v)*(n_e)_y )
      //   * Kpp = 0 (if u,v ~ P_k, p ~ P_{k-1})!!
      //
      // The DenseSubMatrix.reposition () member takes the
      // (row_offset, column_offset, row_size, column_size).
      //
      // Similarly, the DenseSubVector.reposition () member
      // takes the (row_offset, row_size)

      matrix_reposition(Ke_uu, Ke_uv, Ke_vu, Ke_vv,
			u_var, v_var, n_dofs_u, n_dofs_u, n_dofs_v, n_dofs_v);

      Fe_u.reposition (u_var*n_dofs_u, n_dofs_u);
      Fe_v.reposition (v_var*n_dofs_u, n_dofs_v);

      // Now we will build the element interior matrix.  This involves
      // a double loop to integrate the test functions (i) against
      // the trial functions (j).
      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
        {
          // Assemble the u-velocity row
          // uu coupling
          for (unsigned int i=0; i<n_dofs_u; i++)
            for (unsigned int j=0; j<n_dofs_u; j++)
              Ke_uu(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]);


          // Assemble the v-velocity row
          // vv coupling
          for (unsigned int i=0; i<n_dofs_v; i++)
            for (unsigned int j=0; j<n_dofs_v; j++)
              Ke_vv(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]);

        } // end of the quadrature point qp-loop


      // Now we deal with the contributions of the faces of current element
      for (auto side : elem->side_index_range())
        {
	  // If current side is a boundary face:
	  // We consider Dirichlet bc imposed via the interior penalty method
	  // The following loops over the sides of the element.
	  // If the element has no neighbor on a side then that
	  // side MUST live on a boundary of the domain.
	  if (elem->neighbor_ptr(side) == libmesh_nullptr)
	    {
              // Pointer to the element face
              fe_u_elem_face->reinit(elem, side);

              std::unique_ptr<const Elem> elem_side (elem->build_side_ptr(side));
              // h element dimension to compute the interior penalty penalty parameter
              const unsigned int elem_b_order = static_cast<unsigned int> (fe_u_elem_face->get_order());
              const double h_elem = elem->volume()/elem_side->volume() * 1./pow(elem_b_order, 2.);

              for (unsigned int qp=0; qp<qface.n_points(); qp++)
                {
		  // x,y coordinates of quadrature point
                  const Number xf = qface_points[qp](0);
                  const Number yf = qface_points[qp](1);

		  // Set u = 1 on the top boundary, 0 everywhere else
		  const Number u_bc_value = (yf > .9999) ? 1. : 0.;
		  // Set v = 1 on the right boundary, 0 everywhere else
		  const Number v_bc_value = (xf > .9999) ? 1. : 0.;

		  for (unsigned int i=0; i<n_dofs_u; i++) // Warning: assuming n_dofs_u==n_dofs_v !!
                    {
                      // Matrix contribution
                      for (unsigned int j=0; j<n_dofs_u; j++) // Warning: assuming n_dofs_u==n_dofs_v !!
                        {
                          // stability
                          Ke_uu(i,j) += JxW_face[qp] * penalty/h_elem * phi_face[i][qp] * phi_face[j][qp];
                          Ke_vv(i,j) += JxW_face[qp] * penalty/h_elem * phi_face[i][qp] * phi_face[j][qp];

                          // consistency
                          Ke_uu(i,j) -=
                            JxW_face[qp] *
                            (phi_face[i][qp] * (dphi_face[j][qp]*qface_normals[qp]) +
                             phi_face[j][qp] * (dphi_face[i][qp]*qface_normals[qp]));
                          Ke_vv(i,j) -=
                            JxW_face[qp] *
                            (phi_face[i][qp] * (dphi_face[j][qp]*qface_normals[qp]) +
                             phi_face[j][qp] * (dphi_face[i][qp]*qface_normals[qp]));
                        }

                      // RHS contributions

                      // stability
                      Fe_u(i) += JxW_face[qp] * u_bc_value * penalty/h_elem * phi_face[i][qp];
                      Fe_v(i) += JxW_face[qp] * v_bc_value * penalty/h_elem * phi_face[i][qp];

                      // consistency
                      Fe_u(i) -= JxW_face[qp] * dphi_face[i][qp] * (u_bc_value*qface_normals[qp]);
                      Fe_v(i) -= JxW_face[qp] * dphi_face[i][qp] * (v_bc_value*qface_normals[qp]);
                    }
                }
            }

          // If the element is not on a boundary of the domain
          // we loop over his neighbors to compute the element
          // and neighbor boundary matrix contributions
          else
            {
              // Store a pointer to the neighbor we are currently
              // working on.
              const Elem * neighbor = elem->neighbor_ptr(side);

              // Get the global id of the element and the neighbor
              const unsigned int elem_id = elem->id();
              const unsigned int neighbor_id = neighbor->id();

              // If the neighbor has the same h level and is active
              // perform integration only if our global id is bigger than our neighbor id.
              // We don't want to compute twice the same contributions.
              // If the neighbor has a different h level perform integration
              // only if the neighbor is at a lower level.
              if ((neighbor->active() &&
                   (neighbor->level() == elem->level()) &&
                   (elem_id < neighbor_id)) ||
                  (neighbor->level() < elem->level()))
                {
                  // Pointer to the element side
                  std::unique_ptr<const Elem> elem_side (elem->build_side_ptr(side));

                  // h dimension to compute the interior penalty penalty parameter
                  const unsigned int elem_b_order = static_cast<unsigned int>(fe_u_elem_face->get_order());
                  const unsigned int neighbor_b_order = static_cast<unsigned int>(fe_u_neighbor_face->get_order());
                  const double side_order = (elem_b_order + neighbor_b_order)/2.;
                  const double h_elem = (elem->volume()/elem_side->volume()) * 1./pow(side_order,2.);

                  // The quadrature point locations on the neighbor side
                  std::vector<Point> qface_neighbor_point;

                  // The quadrature point locations on the element side
                  std::vector<Point > qface_point;

                  // Reinitialize shape functions on the element side
                  fe_u_elem_face->reinit(elem, side);

                  // Get the physical locations of the element quadrature points
                  qface_point = fe_u_elem_face->get_xyz();

                  // Find their locations on the neighbor
                  unsigned int side_neighbor = neighbor->which_neighbor_am_i(elem);
                  if (refinement_type == "p")
                    fe_u_neighbor_face->side_map (neighbor,
                                                elem_side.get(),
                                                side_neighbor,
                                                qface.get_points(),
                                                qface_neighbor_point);
                  else
                    FEInterface::inverse_map (elem->dim(),
                                              fe_u->get_fe_type(),
                                              neighbor,
                                              qface_point,
                                              qface_neighbor_point);

                  // Calculate the neighbor element shape functions at those locations
                  fe_u_neighbor_face->reinit(neighbor, &qface_neighbor_point);

                  // Get the degree of freedom indices for the
                  // neighbor.  These define where in the global
                  // matrix this neighbor will contribute to.
                  std::vector<dof_id_type> neighbor_dof_indices;
		  std::vector<dof_id_type> neighbor_dof_indices_u;
		  std::vector<dof_id_type> neighbor_dof_indices_v;

                  dof_map.dof_indices (neighbor, neighbor_dof_indices);
		  dof_map.dof_indices (neighbor, neighbor_dof_indices_u, u_var);
		  dof_map.dof_indices (neighbor, neighbor_dof_indices_v, v_var);

                  const unsigned int n_neighbor_dofs = neighbor_dof_indices.size();
		  const unsigned int n_neighbor_dofs_u = neighbor_dof_indices_u.size();
		  const unsigned int n_neighbor_dofs_v = neighbor_dof_indices_v.size();

		  // Zero the element and neighbor side matrix before
                  // summing them.  We use the resize member here because
                  // the number of degrees of freedom might have changed from
                  // the last element or neighbor.
                  // Note that Kne and Ken are not square matrices if neighbor
                  // and element have a different p level
                  Kn.resize (n_neighbor_dofs, n_neighbor_dofs);
                  Ken.resize (n_dofs, n_neighbor_dofs);
                  Kne.resize (n_neighbor_dofs, n_dofs);

		  matrix_reposition(Kn_uu, Kn_uv, Kn_vu, Kn_vv, u_var, v_var,
				    n_neighbor_dofs_u, n_neighbor_dofs_u, n_neighbor_dofs_v, n_neighbor_dofs_v);
		  matrix_reposition(Ken_uu, Ken_uv, Ken_vu, Ken_vv, u_var, v_var,
				    n_dofs_u, n_neighbor_dofs_u, n_dofs_v, n_neighbor_dofs_v);
		  matrix_reposition(Kne_uu, Kne_uv, Kne_vu, Kne_vv, u_var, v_var,
				    n_neighbor_dofs_u, n_dofs_u, n_neighbor_dofs_v, n_dofs_v);


		  // Now we will build the element and neighbor
                  // boundary matrices.  This involves
                  // a double loop to integrate the test functions
                  // (i) against the trial functions (j).
		  //
		  // WARNING: At this moment, we suppose for u and v
		  // the samen FE and the same #dofs in all
		  // elements. In particular,r¡¡¡ This invalidates
		  // p-adaptivity !!!
		  //
                  for (unsigned int qp=0; qp<qface.n_points(); qp++)
                    {
                      // Kee Matrix. Integrate the element test function i
                      // against the element test function j
                      for (unsigned int i=0; i<n_dofs_u; i++) // Warning: assuming n_dofs_u==n_dofs_v !!
                        {
                          for (unsigned int j=0; j<n_dofs_u; j++) // Warning: assuming n_dofs_u==n_dofs_v !!
                            {
                              // consistency
                              Ke_uu(i,j) -=
                                0.5 * JxW_face[qp] *
                                (phi_face[j][qp]*(qface_normals[qp]*dphi_face[i][qp]) +
                                 phi_face[i][qp]*(qface_normals[qp]*dphi_face[j][qp]));
                              Ke_vv(i,j) -=
                                0.5 * JxW_face[qp] *
                                (phi_face[j][qp]*(qface_normals[qp]*dphi_face[i][qp]) +
                                 phi_face[i][qp]*(qface_normals[qp]*dphi_face[j][qp]));

                              // stability
                              Ke_uu(i,j) += JxW_face[qp] * penalty/h_elem * phi_face[j][qp]*phi_face[i][qp];
                              Ke_vv(i,j) += JxW_face[qp] * penalty/h_elem * phi_face[j][qp]*phi_face[i][qp];
                            }
                        }

                      // Knn Matrix. Integrate the neighbor test function i
                      // against the neighbor test function j
                      for (unsigned int i=0; i<n_neighbor_dofs_u; i++) // Warning: assuming n_neighbor_dofs_u==n_neighbor_dofs_v !!
                        {
                          for (unsigned int j=0; j<n_neighbor_dofs_u; j++) // Warning: assuming n_neighbor_dofs_u==n_neighbor_dofs_v !!
                            {
                              // consistency
                              Kn_uu(i,j) +=
                                0.5 * JxW_face[qp] *
                                (phi_neighbor_face[j][qp]*(qface_normals[qp]*dphi_neighbor_face[i][qp]) +
                                 phi_neighbor_face[i][qp]*(qface_normals[qp]*dphi_neighbor_face[j][qp]));
                              Kn_vv(i,j) +=
                                0.5 * JxW_face[qp] *
                                (phi_neighbor_face[j][qp]*(qface_normals[qp]*dphi_neighbor_face[i][qp]) +
                                 phi_neighbor_face[i][qp]*(qface_normals[qp]*dphi_neighbor_face[j][qp]));

                              // stability
                              Kn_uu(i,j) +=
                                JxW_face[qp] * penalty/h_elem * phi_neighbor_face[j][qp]*phi_neighbor_face[i][qp];
                              Kn_vv(i,j) +=
                                JxW_face[qp] * penalty/h_elem * phi_neighbor_face[j][qp]*phi_neighbor_face[i][qp];
                            }
                        }

                      // Kne Matrix. Integrate the neighbor test function i
                      // against the element test function j
                      for (unsigned int i=0; i<n_neighbor_dofs_u; i++) // Warning: assuming n_neighbor_dofs_u==n_neighbor_dofs_v !!
                        {
                          for (unsigned int j=0; j<n_dofs_u; j++) // Warning: assuming n_dofs_u==n_dofs_v !!
                            {
                              // consistency
                              Kne_uu(i,j) +=
                                0.5 * JxW_face[qp] *
                                (phi_neighbor_face[i][qp]*(qface_normals[qp]*dphi_face[j][qp]) -
                                 phi_face[j][qp]*(qface_normals[qp]*dphi_neighbor_face[i][qp]));
                              Kne_vv(i,j) +=
                                0.5 * JxW_face[qp] *
                                (phi_neighbor_face[i][qp]*(qface_normals[qp]*dphi_face[j][qp]) -
                                 phi_face[j][qp]*(qface_normals[qp]*dphi_neighbor_face[i][qp]));

                              // stability
                              Kne_uu(i,j) -= JxW_face[qp] * penalty/h_elem * phi_face[j][qp]*phi_neighbor_face[i][qp];
                              Kne_vv(i,j) -= JxW_face[qp] * penalty/h_elem * phi_face[j][qp]*phi_neighbor_face[i][qp];
                            }
                        }

                      // Ken Matrix. Integrate the element test function i
                      // against the neighbor test function j
                      for (unsigned int i=0; i<n_dofs_u; i++) // Warning: assuming n_dofs_u==n_dofs_v !!
                        {
                          for (unsigned int j=0; j<n_neighbor_dofs_u; j++) // Warning: assuming n_neighbor_dofs_u==n_neighbor_dofs_v !!
                            {
                              // consistency
                              Ken_uu(i,j) +=
                                0.5 * JxW_face[qp] *
                                (phi_neighbor_face[j][qp]*(qface_normals[qp]*dphi_face[i][qp]) -
                                 phi_face[i][qp]*(qface_normals[qp]*dphi_neighbor_face[j][qp]));
                              Ken_vv(i,j) +=
                                0.5 * JxW_face[qp] *
                                (phi_neighbor_face[j][qp]*(qface_normals[qp]*dphi_face[i][qp]) -
                                 phi_face[i][qp]*(qface_normals[qp]*dphi_neighbor_face[j][qp]));

                              // stability
                              Ken_uu(i,j) -= JxW_face[qp] * penalty/h_elem * phi_face[i][qp]*phi_neighbor_face[j][qp];
                              Ken_vv(i,j) -= JxW_face[qp] * penalty/h_elem * phi_face[i][qp]*phi_neighbor_face[j][qp];
                            }
                        }
                    }

                  // The element and neighbor boundary matrix are now built
                  // for this side.  Add them to the global matrix
                  // The SparseMatrix::add_matrix() members do this for us.
                  stokesdg_system.matrix->add_matrix(Kn, neighbor_dof_indices);
                  stokesdg_system.matrix->add_matrix(Ken, dof_indices, neighbor_dof_indices);
                  stokesdg_system.matrix->add_matrix(Kne, neighbor_dof_indices, dof_indices);
                }
            }
        }
      // The element interior matrix and right-hand-side are now built
      // for this element.  Add them to the global matrix and
      // right-hand-side vector.  The SparseMatrix::add_matrix()
      // and NumericVector::add_vector() members do this for us.
      stokesdg_system.matrix->add_matrix(Ke, dof_indices);
      stokesdg_system.rhs->add_vector(Fe, dof_indices);
    }

  libMesh::out << "done" << std::endl;
}



int main (int argc, char** argv)
{
  LibMeshInit init(argc, argv);

  // This example requires a linear solver package.
  libmesh_example_requires(libMesh::default_solver_package() != INVALID_SOLVER_PACKAGE,
                           "--enable-petsc, --enable-trilinos, or --enable-eigen");

  // Skip adaptive examples on a non-adaptive libMesh build
#ifndef LIBMESH_ENABLE_AMR
  libmesh_example_requires(false, "--enable-amr");
#else

  //Parse the input file
  GetPot input_file("stokes_dg_01.in");

  //Read in parameters from the input file
  const unsigned int adaptive_refinement_steps = input_file("max_adaptive_r_steps", 3);
  const unsigned int uniform_refinement_steps  = input_file("uniform_h_r_steps", 3);
  const Real refine_fraction                   = input_file("refine_fraction", 0.5);
  const Real coarsen_fraction                  = input_file("coarsen_fraction", 0.);
  const unsigned int max_h_level               = input_file("max_h_level", 10);
  const std::string refinement_type            = input_file("refinement_type","p");
  Order p_order                                = static_cast<Order>(input_file("p_order", 1));
  const std::string element_type               = input_file("element_type", "tensor");
  const Real penalty                           = input_file("ip_penalty", 10.);
  const bool singularity                       = input_file("singularity", true);
  const unsigned int dim                       = input_file("dimension", 3);

  // Skip higher-dimensional examples on a lower-dimensional libMesh build
  libmesh_example_requires(dim <= LIBMESH_DIM, "2D/3D support");


  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());
  MeshTools::Generation::build_square (mesh,
                                       15, 15,
                                       0., 1.,
                                       0., 1.,
                                       QUAD9);

  // Use triangles if the config file says so
  if (element_type == "simplex")
    MeshTools::Modification::all_tri(mesh);

  // Mesh Refinement object
  MeshRefinement mesh_refinement(mesh);
  mesh_refinement.refine_fraction() = refine_fraction;
  mesh_refinement.coarsen_fraction() = coarsen_fraction;
  mesh_refinement.max_h_level() = max_h_level;

  // Do uniform refinement
  for (unsigned int rstep=0; rstep<uniform_refinement_steps; rstep++)
    mesh_refinement.uniformly_refine(1);

  // Crate an equation system object
  EquationSystems equation_system (mesh);

  // Set parameters for the equation system and the solver
  equation_system.parameters.set<Real>("linear solver tolerance") = TOLERANCE * TOLERANCE;
  equation_system.parameters.set<unsigned int>("linear solver maximum iterations") = 1000;
  equation_system.parameters.set<Real>("penalty") = penalty;
  equation_system.parameters.set<bool>("singularity") = singularity;
  equation_system.parameters.set<std::string>("refinement") = refinement_type;

  // Create a system named stokesdg
  LinearImplicitSystem & stokesdg_system = equation_system.add_system<LinearImplicitSystem> ("StokesDG");

  // Add a variable "u" to "stokesdg" using the p_order specified in the config file
  if (on_command_line("element_type"))
    {
      std::string fe_str =
        command_line_value(std::string("element_type"),
                           std::string("MONOMIAL"));

      if (fe_str != "MONOMIAL" || fe_str != "XYZ")
        libmesh_error_msg("Error: This example must be run with MONOMIAL or XYZ element types.");

      stokesdg_system.add_variable ("u", p_order, Utility::string_to_enum<FEFamily>(fe_str));
      stokesdg_system.add_variable ("v", p_order, Utility::string_to_enum<FEFamily>(fe_str));
    }
  else {
    stokesdg_system.add_variable ("u", p_order, MONOMIAL);
    stokesdg_system.add_variable ("v", p_order, MONOMIAL);
    }

  // Give the system a pointer to the matrix assembly function
  stokesdg_system.attach_assemble_function (assemble_stokesdg);

  // Initialize the data structures for the equation system
  equation_system.init();

  // A refinement loop.
  for (unsigned int rstep=0; rstep<adaptive_refinement_steps; ++rstep)
    {
      libMesh::out << "  Beginning Solve " << rstep << std::endl;
      libMesh::out << "Number of elements: " << mesh.n_elem() << std::endl;

      // Solve the system
      stokesdg_system.solve();

      libMesh::out << "System has: "
                   << equation_system.n_active_dofs()
                   << " degrees of freedom."
                   << std::endl;

      libMesh::out << "Linear solver converged at step: "
                   << stokesdg_system.n_linear_iterations()
                   << ", final residual: "
                   << stokesdg_system.final_linear_residual()
                   << std::endl;


      // Possibly refine the mesh
      if (rstep+1 < adaptive_refinement_steps)
        {
          // The ErrorVector is a particular StatisticsVector
          // for computing error information on a finite element mesh.
          ErrorVector error;

          // The discontinuity error estimator
          // evaluate the jump of the solution
          // on elements faces
          DiscontinuityMeasure error_estimator;
          error_estimator.estimate_error(stokesdg_system, error);

          // Take the error in error and decide which elements will be coarsened or refined
          mesh_refinement.flag_elements_by_error_fraction(error);
          if (refinement_type == "p")
            mesh_refinement.switch_h_to_p_refinement();
          if (refinement_type == "hp")
            mesh_refinement.add_p_to_h_refinement();

          // Refine and coarsen the flagged elements
          mesh_refinement.refine_and_coarsen_elements();
          equation_system.reinit();
        }
    }

  // Write out the solution
  // After solving the system write the solution
  // to a ExodusII-formatted plot file.
#ifdef LIBMESH_HAVE_EXODUS_API
  ExodusII_IO (mesh).write_discontinuous_exodusII("stokes_dg.e", equation_system);
#endif

#endif // #ifndef LIBMESH_ENABLE_AMR

  // All done.
  return 0;
}
