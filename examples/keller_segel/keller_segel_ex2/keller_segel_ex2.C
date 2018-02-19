// The libMesh Finite Element Library.
// Copyright (C) 2002-2017 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

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



// <h1>Keller-Segel Example 1 - Solving a Keller-Segel Linear System in Parallel</h1>
// \author Rafa Rodríguez Galván
// \date 2018
//
// This example solves a Keller-Segel system.  The initial condition
// is given, and the solution is advanced in time with a standard
// backward (implicit) Euler time-stepping strategy.

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <sstream>
#include <math.h>

// Basic include file needed for the mesh functionality.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h" // For build_square()
// #include "libmesh/mesh_refinement.h"
#include "libmesh/gmv_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/fe_base.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/getpot.h"

// This example will solve a linear transient system,
// so we need to include the TransientLinearImplicitSystem definition.
#include "libmesh/linear_implicit_system.h"
#include "libmesh/transient_system.h"
#include "libmesh/vector_value.h"

// The definition of a geometric element
#include "libmesh/elem.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// This function will assemble the system matrix and right-hand-side
// for u at each time step.  Note that since the system is linear we
// technically do not need to assmeble the matrix at each time step,
// but we will anyway.  In subsequent examples we will employ adaptive
// mesh refinement, and with a changing mesh it will be necessary to
// rebuild the system matrix.
void assemble_ks_u (EquationSystems & es,
		    const std::string & system_name);
// This function will assemble the system matrix and right-hand-side
// for v at each time step.
void assemble_ks_v (EquationSystems & es,
		    const std::string & system_name);

// Function prototype.  This function will initialize the system.
// Initialization functions are optional for systems.  They allow
// you to specify the initial values of the solution.  If an
// initialization function is not provided then the default (0)
// solution is provided.
void init_ks_u (EquationSystems & es,
		const std::string & system_name);
void init_ks_v (EquationSystems & es,
		const std::string & system_name);

// Initial condition for u
Number initial_value_u(const Point& p,
			const Parameters&,
			const std::string&,
			const std::string&)
{
  Number x=p(0), y=p(1);
  return  1.15*exp(-x*x-y*y)*(4-x*x)*(4-x*x)*(4-y*y)*(4-y*y);
  // return 1;
}

// Initial condition for v
Number initial_value_v(const Point& p,
			const Parameters&,
			const std::string&,
			const std::string&)
{
  Number x=p(0), y=p(1);
  return 0.55*exp(-x*x-y*y)*(4-x*x)*(4-x*x)*(4-y*y)*(4-y*y);
  // return 1;
}

// Begin the main program
// ==============================
int main (int argc, char ** argv)
// ==============================

{
  // Initialize libMesh.
  LibMeshInit init (argc, argv);

  // This example requires a linear solver package.
  libmesh_example_requires(libMesh::default_solver_package() != INVALID_SOLVER_PACKAGE,
                           "--enable-petsc, --enable-trilinos, or --enable-eigen");

  // This example requires Adaptive Mesh Refinement support - although
  // it only refines uniformly, the refinement code used is the same
  // underneath
#ifndef LIBMESH_ENABLE_AMR
  libmesh_example_requires(false, "--enable-amr");
#else

  // Skip this 2D example if libMesh was compiled as 1D-only.
  libmesh_example_requires(2 <= LIBMESH_DIM, "2D support");

  // Parse options, and select a file for more options
  GetPot options(argc, argv);
  options.parse_command_line(argc, argv);
  std::string input_file = options("infile", std::string("keller_segel_ex2.in"));

  // Parse the input file, but keep more precedence to console arguments
  options.parse_input_file(input_file);
  options.parse_command_line(argc, argv);

  // Read in parameters from the input file
  const unsigned int n_mesh_intervals = options("nx", 20);
  const Real dt                       = options("dt", 0.0001);
  const unsigned int n_time_steps     = options("nt", 5);
  const unsigned int save_n_steps     = options("save_n_steps", 10);
  const unsigned int fe_order         = options("order", 1);
  std::string fe_family_str           = options("family", std::string("LAGRANGE"));
  FEFamily fe_family = Utility::string_to_enum<FEFamily>(fe_family_str);

  options.print(std::cout);

  // Time scheme
  // u_t - c_u1 \Delta u^m+1 + c_u2 \div(u^{m+r1} \grad v^{m+r2})
  // v_t - c_v2 \Delta v^m+1 + c_v2 v^{m+r3} + c_v3 u{m+r4}
  // scheme=0: ri=0, i=1,2,3,4
  // scheme=1: r3=1, r1=r2=r4=0
  // scheme=2: r1=r3=r4=1, r2=0
  // scheme=3: r1=r2=r3=1, r4=0
  const unsigned int time_scheme = options("scheme", 0);

  // Keller-Segel parameters
  const Real c_u1 = options("c_u1", 1.0);
  const Real c_u2 = options("c_u2", 0.2);
  const Real c_v1 = options("c_v1", 1.0);
  const Real c_v2 = options("c_v2", 0.1);
  const Real c_v3 = options("c_v3", 1.0);

  // Read the mesh from file.  This is the coarse mesh that will be used
  // in example 10 to demonstrate adaptive mesh refinement.  Here we will
  // simply read it in and uniformly refine it 5 times before we compute
  // with it.
  //
  // Create a mesh object, with dimension to be overridden later,
  // distributed across the default MPI communicator.
  Mesh mesh(init.comm());

  // MeshTools::Generation::build_cube (mesh,
  MeshTools::Generation::build_square (mesh,
                                       n_mesh_intervals, n_mesh_intervals,
                                       -2., 2.,
                                       -2., 2.,
                                       TRI6);

  // mesh.read ("mesh.xda");

  // Create a MeshRefinement object to handle refinement of our mesh.
  // This class handles all the details of mesh refinement and coarsening.
  // MeshRefinement mesh_refinement (mesh);

  // Uniformly refine the mesh 5 times.  This is the
  // first time we use the mesh refinement capabilities
  // of the library.
  // mesh_refinement.uniformly_refine (5);

  // Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);

  // Add a transient system object named "Keller-Segel.u".
  TransientLinearImplicitSystem & system_u =
    equation_systems.add_system<TransientLinearImplicitSystem> ("Keller-Segel.u");

  // Add a transient system object named "Keller-Segel.v".
  TransientLinearImplicitSystem & system_v =
    equation_systems.add_system<TransientLinearImplicitSystem> ("Keller-Segel.v");

  // Adds the variable "u" to "Keller-Segel.u", using first-order approximation.
  system_u.add_variable ("u", static_cast<Order>(fe_order), fe_family);

  // Adds the variable "v" to "Keller-Segel.v", using first-order approximation.
  system_v.add_variable ("v", static_cast<Order>(fe_order), fe_family);

  // Give the system a pointer to the matrix assembly for u,v
  // and initialization functions.
  system_u.attach_assemble_function (assemble_ks_u);
  system_v.attach_assemble_function (assemble_ks_v);
  system_u.attach_init_function (init_ks_u);
  system_v.attach_init_function (init_ks_v);

  // Initialize the data structures for the equation system.
  equation_systems.init();

  // Prints information about the system to the screen.
  equation_systems.print_info();
  {
    std::ostringstream out;

    out << std::setw(2)
	<< std::right
	<< "time="
	<< std::fixed
	<< std::setw(9)
	<< std::setprecision(6)
	<< std::setfill('0')
	<< std::left
	<< system_u.time << " (" << system_v.time << ")"

	<<  "...";


      out << std::scientific
	  << std::setprecision(9)
	  << std::endl << "  max(u): " << system_u.solution->max()
	  << ",  min(u): " << system_u.solution->min()
	  << std::endl << "  max(v): " << system_v.solution->max()
	  << ",  min(v): " << system_v.solution->min();

    libMesh::out << out.str() << std::endl;
  }

  // Write out the initial conditions.
#ifdef LIBMESH_HAVE_EXODUS_API
  // If Exodus is available, we'll write all time_steps to the same file
  // rather than one file per timestep.
  std::string exodus_filename = "keller_segel_ex2.e";
  ExodusII_IO(mesh).write_equation_systems (exodus_filename, equation_systems);
#else
  GMVIO(mesh).write_equation_systems ("out_000.gmv", equation_systems);
#endif

  // The Keller-Segel system requires that we specify
  // some parameter to pass them to the assemble function.
  equation_systems.parameters.set<unsigned int>("scheme") = time_scheme;
  equation_systems.parameters.set<Real>("c_u1") = c_u1;
  equation_systems.parameters.set<Real>("c_u2") = c_u2;
  equation_systems.parameters.set<Real>("c_v1") = c_v1;
  equation_systems.parameters.set<Real>("c_v2") = c_v2;
  equation_systems.parameters.set<Real>("c_v3") = c_v3;

  // Solve the system "Keller-Segel".  This will be done by
  // looping over the specified time interval and calling the
  // solve() member at each time step.  This will assemble the
  // system and call the linear solver.
  system_u.time = 0.;
  system_v.time = 0.;

  for (unsigned int t_step = 0; t_step < n_time_steps; t_step++)
    {
      // Incremenet the time counter, set the time and the
      // time step size as parameters in the EquationSystem.
      system_u.time += dt;
      system_v.time += dt;

      equation_systems.parameters.set<Real> ("time") = system_u.time;
      equation_systems.parameters.set<Real> ("dt")   = dt;

      // A pretty update message
      libMesh::out << " Solving time step ";

      // Do fancy zero-padded formatting of the current time.
      {
        std::ostringstream out;

        out << std::setw(2)
            << std::right
            << t_step
            << ", time="
            << std::fixed
            << std::setw(9)
            << std::setprecision(6)
            << std::setfill('0')
            << std::left
            << system_u.time << " (" << system_v.time << ")"

            <<  "...";


	if ((t_step+1)%save_n_steps == 0 || t_step+1==n_time_steps)
	  out << std::scientific
	      << std::setprecision(9)
	      << std::endl << "  max(u): " << system_u.solution->max()
	      << ",  min(u): " << system_u.solution->min()
	      << std::endl << "  max(v): " << system_v.solution->max()
	      << ",  min(v): " << system_v.solution->min();

	libMesh::out << out.str() << std::endl;
      }

      // At this point we need to update the old solution vector.  The
      // old solution vector will be the current solution vector from
      // the previous time step.  We will do this by extracting the
      // system from the EquationSystems object and using vector
      // assignment.  Since only TransientSystems (and systems derived
      // from them) contain old solutions we need to specify the
      // system type when we ask for it.
      *system_u.old_local_solution = *system_u.current_local_solution;
      *system_v.old_local_solution = *system_v.current_local_solution;

      // Assemble & solve the linear system for u
      equation_systems.get_system("Keller-Segel.u").solve();
      // Assemble & solve the linear system for v
      equation_systems.get_system("Keller-Segel.v").solve();

      // Output evey nsteps time_steps to file.
      if ((t_step+1)%save_n_steps == 0)
        {

#ifdef LIBMESH_HAVE_EXODUS_API
          ExodusII_IO exo(mesh);
          exo.append(true);
          exo.write_timestep (exodus_filename, equation_systems, t_step+1, system_u.time);
#else
          std::ostringstream file_name;

          file_name << "out_"
                    << std::setw(3)
                    << std::setfill('0')
                    << std::right
                    << t_step+1
                    << ".gmv";


          GMVIO(mesh).write_equation_systems (file_name.str(),
                                              equation_systems);
#endif
        }
    }
#endif // #ifdef LIBMESH_ENABLE_AMR

  // All done.
  return 0;
}

// We now define the function which provides the initialization
// routines for the "Keller-Segel" system.  This handles things like
// setting initial conditions and boundary conditions.
void init_ks_u (EquationSystems & es,
              const std::string & libmesh_dbg_var(system_name))
{
  // It is a good idea to make sure we are initializing
  // the proper system.
  libmesh_assert_equal_to (system_name, "Keller-Segel.u");

  // Get a reference to the Keller-Segel system object.
  TransientLinearImplicitSystem & system_u =
    es.get_system<TransientLinearImplicitSystem>("Keller-Segel.u");

  // Project initial conditions at time 0
  es.parameters.set<Real> ("time") = system_u.time = 0;

  system_u.project_solution(initial_value_u, libmesh_nullptr, es.parameters);
}

// We now define the function which provides the initialization for v
void init_ks_v (EquationSystems & es,
              const std::string & libmesh_dbg_var(system_name))
{
  // It is a good idea to make sure we are initializing
  // the proper system.
  libmesh_assert_equal_to (system_name, "Keller-Segel.v");

  // Get a reference to the Keller-Segel system object.
  TransientLinearImplicitSystem & system_v =
    es.get_system<TransientLinearImplicitSystem>("Keller-Segel.v");

  // Project initial conditions at time 0
  es.parameters.set<Real> ("time") = system_v.time = 0;

  system_v.project_solution(initial_value_v, libmesh_nullptr, es.parameters);
}


// Now we define the assemble function which will be used
// by the EquationSystems object at each timestep to assemble
// the linear system for solution.
void assemble_ks_u (EquationSystems & es,
                  const std::string & libmesh_dbg_var(system_name))
{
#ifdef LIBMESH_ENABLE_AMR
  // It is a good idea to make sure we are assembling
  // the proper system.
  libmesh_assert_equal_to (system_name, "Keller-Segel.u");

  // Get a constant reference to the mesh object.
  const MeshBase & mesh = es.get_mesh();

  // The dimension that we are running
  const unsigned int dim = mesh.mesh_dimension();

  // Get a reference to the Keller-Segel.u system object.
  TransientLinearImplicitSystem & system =
    es.get_system<TransientLinearImplicitSystem> ("Keller-Segel.u");

  // Get a reference to the Keller-Segel.v system object.
  TransientLinearImplicitSystem & system_v =
    es.get_system<TransientLinearImplicitSystem> ("Keller-Segel.v");

  // Get a constant reference to the Finite Element type for u (the first
  // (and only) variable in the system).
  FEType fe_type = system.variable_type(0);

  // Get a constant reference to the Finite Element type for v and
  // assert it is equal to FE type for u
  FEType fe_type_v = system_v.variable_type(0);
  assert(fe_type_v == fe_type);

  // Build a Finite Element object of the specified type.  Since the
  // FEBase::build() member dynamically creates memory we will
  // store the object as a UniquePtr<FEBase>.  This can be thought
  // of as a pointer that will clean up after itself.
  UniquePtr<FEBase> fe      (FEBase::build(dim, fe_type));
  UniquePtr<FEBase> fe_face (FEBase::build(dim, fe_type));

  // A Gauss quadrature rule for numerical integration.
  // Let the FEType object decide what order rule is appropriate.
  QGauss qrule (dim,   fe_type.default_quadrature_order());
  QGauss qface (dim-1, fe_type.default_quadrature_order());

  // Tell the finite element object to use our quadrature rule.
  fe->attach_quadrature_rule      (&qrule);
  fe_face->attach_quadrature_rule (&qface);

  // Here we define some references to cell-specific data that
  // will be used to assemble the linear system.  We will start
  // with the element Jacobian * quadrature weight at each integration point.
  const std::vector<Real>& JxW      = fe->get_JxW();

  // The element shape functions evaluated at the quadrature points.
  const std::vector<std::vector<Real>>& phi = fe->get_phi();

  // The element shape function gradients evaluated at the quadrature
  // points.
  const std::vector<std::vector<RealGradient>>& dphi = fe->get_dphi();

  // A reference to the DofMap object for this system. The DofMap
  // object handles the index translation from node and element
  // numbers to degree of freedom numbers.
  const DofMap & dof_map = system.get_dof_map();

  // Define data structures to contain the element matrix and
  // right-hand-side vector contribution.  Following basic finite
  // element terminology we will denote these "Ke" and "Fe".
  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  // This vector will hold the degree of freedom indices for the
  // element. These define where in the global system the element
  // degrees of freedom get mapped.
  std::vector<dof_id_type> dof_indices;

  // Extract parameters
  const Real dt = es.parameters.get<Real> ("dt");
  const unsigned int scheme =
    es.parameters.get<unsigned int> ("scheme");
  const Real c_u1 = es.parameters.get<Real> ("c_u1");
  const Real c_u2 = es.parameters.get<Real> ("c_u2");

  // Now we will loop over all the elements in the mesh that live on
  // the local processor. We will compute the element matrix and
  // right-hand-side contribution.  Since the mesh will be refined we
  // want to only consider the ACTIVE elements, hence we use a variant
  // of the active_elem_iterator.
  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

  for ( ; el != end_el; ++el)
    {
      // Store a pointer to the element we are currently
      // working on.  This allows for nicer syntax later.
      const Elem * elem = *el;

      // Get the degree of freedom indices for the
      // current element.  These define where in the global
      // matrix and right-hand-side this element will
      // contribute to.
      dof_map.dof_indices (elem, dof_indices);

      // Compute the element-specific data for the current
      // element.  This involves computing the location of the
      // quadrature points (q_point) and the shape functions
      // (phi, dphi) for the current element.
      fe->reinit (elem);

      // Zero the element matrix and right-hand side before
      // summing them.  We use the resize member here because
      // the number of degrees of freedom might have changed from
      // the last element.  Note that this will be the case if the
      // element type is different (i.e. the last element was a
      // triangle, now we are on a quadrilateral).
      Ke.resize (dof_indices.size(),
                 dof_indices.size());

      Fe.resize (dof_indices.size());

      // Now we will build the element matrix and right-hand-side.
      // Constructing the RHS requires the solution and its
      // gradient from the previous timestep.  This myst be
      // calculated at each quadrature point by summing the
      // solution degree-of-freedom values by the appropriate
      // weight functions.
      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
        {
          // Values to hold the old solution & its gradient.
          Number u_old = 0.;
          Gradient grad_u_old;
	  // Previous values of v & its gradient
	  Number v_old = 0.;
	  Gradient grad_v = 0.;

          // Compute the old solution & its gradient.
          for (std::size_t l=0; l<phi.size(); l++)
            {
              u_old += phi[l][qp]*system.old_solution (dof_indices[l]);
              v_old += phi[l][qp]*system_v.old_solution (dof_indices[l]);

              grad_u_old.add_scaled (dphi[l][qp], system.old_solution (dof_indices[l]));
	      // In schemes 0,1,2 grad v is given explictly, in scheme 3 grad v is implicit
	      // Note that in scheme 3 v system must be solved before assembling u
              grad_v.add_scaled (dphi[l][qp],
				 scheme==3 ?
				 system_v.current_solution (dof_indices[l]) :
				 system_v.old_solution (dof_indices[l])
				 );
	      // std::cout << system_v.old_solution(dof_indices[l]) << std::endl;
            }

          // Now compute the element matrix and RHS contributions, for each local dof.
          for (std::size_t i=0; i<phi.size(); i++)
            {
	      // The matrix contribution
	      for (std::size_t j=0; j<phi.size(); j++)
		{
		  Ke(i,j) += JxW[qp]*(
                                      // Time derivative (mass-matrix)
                                      phi[i][qp]*phi[j][qp]
				      // Diffusion term
				      + dt*c_u1 * (dphi[i][qp]*dphi[j][qp])
				      // Convection term (implicit in schemes 2,3)
				      - ( scheme==2 || scheme==3 ?
					  dt*c_u2 * phi[i][qp]*(grad_v*dphi[j][qp])
					  : 0 )
                                      );
                }
	      // The RHS contribution
	      Fe(i) += JxW[qp]*(
				// Mass matrix term
                                u_old*phi[i][qp] +
				// Convection term (explicit in schemes 0, 1)
				+ ( scheme==0 || scheme==1 ?
				    dt*c_u2 * u_old*(grad_v*dphi[i][qp])
				    : 0)
                                );

            }
        }

      // If this assembly program were to be used on an adaptive mesh,
      // we would have to apply any hanging node constraint equations
      dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

      // The element matrix and right-hand-side are now built
      // for this element.  Add them to the global matrix and
      // right-hand-side vector.  The SparseMatrix::add_matrix()
      // and NumericVector::add_vector() members do this for us.
      system.matrix->add_matrix (Ke, dof_indices);
      system.rhs->add_vector    (Fe, dof_indices);
    }

  // That concludes the system matrix assembly routine for u
#endif // #ifdef LIBMESH_ENABLE_AMR
}

// Now we define the assemble function for v
void assemble_ks_v (EquationSystems & es,
		    const std::string & libmesh_dbg_var(system_name))
{
#ifdef LIBMESH_ENABLE_AMR
  // It is a good idea to make sure we are assembling
  // the proper system.
  libmesh_assert_equal_to (system_name, "Keller-Segel.v");

  // Get a constant reference to the mesh object.
  const MeshBase & mesh = es.get_mesh();

  // The dimension that we are running
  const unsigned int dim = mesh.mesh_dimension();

  // Get a reference to the Keller-Segel.u system object.
  TransientLinearImplicitSystem & system =
    es.get_system<TransientLinearImplicitSystem> ("Keller-Segel.v");

  // Get a reference to the Keller-Segel.v system object.
  TransientLinearImplicitSystem & system_u =
    es.get_system<TransientLinearImplicitSystem> ("Keller-Segel.u");

  // Get a constant reference to the Finite Element type for v (the first
  // (and only) variable in the system).
  FEType fe_type = system.variable_type(0);

  // Get a constant reference to the Finite Element type for u and
  // assert it is equal to FE type for u
  FEType fe_type_u = system_u.variable_type(0);
  assert(fe_type_u == fe_type);

  // Build a Finite Element object of the specified type.  Since the
  // FEBase::build() member dynamically creates memory we will
  // store the object as a UniquePtr<FEBase>.  This can be thought
  // of as a pointer that will clean up after itself.
  UniquePtr<FEBase> fe      (FEBase::build(dim, fe_type));
  UniquePtr<FEBase> fe_face (FEBase::build(dim, fe_type));

  // A Gauss quadrature rule for numerical integration.
  // Let the FEType object decide what order rule is appropriate.
  QGauss qrule (dim,   fe_type.default_quadrature_order());
  QGauss qface (dim-1, fe_type.default_quadrature_order());

  // Tell the finite element object to use our quadrature rule.
  fe->attach_quadrature_rule      (&qrule);
  fe_face->attach_quadrature_rule (&qface);

  // Here we define some references to cell-specific data that
  // will be used to assemble the linear system.  We will start
  // with the element Jacobian * quadrature weight at each integration point.
  const std::vector<Real>& JxW      = fe->get_JxW();

  // The element shape functions evaluated at the quadrature points.
  const std::vector<std::vector<Real>>& phi = fe->get_phi();

  // The element shape function gradients evaluated at the quadrature
  // points.
  const std::vector<std::vector<RealGradient>>& dphi = fe->get_dphi();

  // A reference to the DofMap object for this system. The DofMap
  // object handles the index translation from node and element
  // numbers to degree of freedom numbers.
  const DofMap & dof_map = system.get_dof_map();

  // Define data structures to contain the element matrix and
  // right-hand-side vector contribution.  Following basic finite
  // element terminology we will denote these "Ke" and "Fe".
  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  // This vector will hold the degree of freedom indices for the
  // element. These define where in the global system the element
  // degrees of freedom get mapped.
  std::vector<dof_id_type> dof_indices;

  // Extract parameters
  const Real dt = es.parameters.get<Real> ("dt");
  const unsigned int scheme =
    es.parameters.get<unsigned int> ("scheme");
  const Real c_v1 = es.parameters.get<Real> ("c_v1");
  const Real c_v2 = es.parameters.get<Real> ("c_v2");
  const Real c_v3 = es.parameters.get<Real> ("c_v3");

  // Now we will loop over all the elements in the mesh that live on
  // the local processor. We will compute the element matrix and
  // right-hand-side contribution.  Since the mesh will be refined we
  // want to only consider the ACTIVE elements, hence we use a variant
  // of the active_elem_iterator.
  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

  for ( ; el != end_el; ++el)
    {
      // Store a pointer to the element we are currently
      // working on.  This allows for nicer syntax later.
      const Elem * elem = *el;

      // Get the degree of freedom indices for the current element.
      // These define where in the global matrix and right-hand-side
      // this element will contribute to.
      dof_map.dof_indices (elem, dof_indices);

      // Compute the element-specific data for the current element.
      // This involves computing the location of the quadrature points
      // (q_point) and the shape functions (phi, dphi) for the current
      // element.
      fe->reinit (elem);

      // Zero the element matrix and right-hand side before
      // summing them.  We use the resize member here because
      // the number of degrees of freedom might have changed from
      // the last element.  Note that this will be the case if the
      // element type is different (i.e. the last element was a
      // triangle, now we are on a quadrilateral).
      Ke.resize (dof_indices.size(),
                 dof_indices.size());

      Fe.resize (dof_indices.size());

      // Now we will build the element matrix and right-hand-side.
      // Constructing the RHS requires the solution and its
      // gradient from the previous timestep.  This myst be
      // calculated at each quadrature point by summing the
      // solution degree-of-freedom values by the appropriate
      // weight functions.
      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
        {
          // Values to hold the old solution & its gradient.
          Number u_old = 0.;
          Gradient grad_u_old;
	  // Previous values of v & its gradient
	  Number v_old = 0.;
	  Gradient grad_v_old = 0.;

          // Compute the old solution & its gradient.
          for (std::size_t l=0; l<phi.size(); l++)
            {
	      // In schemes 0,1,3 the term "u" is given explictly. In
	      // scheme 2, it is implicit. Note that in scheme 2, u
	      // system must be solved before assembling u
	      u_old += phi[l][qp]*( scheme==2 ?
				    system_u.current_solution(dof_indices[l]) :
				    system_u.old_solution(dof_indices[l])
				   );
	      v_old += phi[l][qp]*system.old_solution(dof_indices[l]);
            }

          // Now compute the element matrix and RHS contributions, for each local dof.
          for (std::size_t i=0; i<phi.size(); i++)
            {
	      // The matrix contribution
	      for (std::size_t j=0; j<phi.size(); j++)
		{
		  Ke(i,j) += JxW[qp]*(
                                      // Time derivative (mass-matrix)
                                      phi[i][qp]*phi[j][qp]
				      // Diffusion term (stiffness matrix)
				      + dt*c_v1 * (dphi[i][qp]*dphi[j][qp])
				      // Reaction v (implicit in schemes 1,2,3)
				      + ( scheme==0 ? 0 :
					  dt*c_v2 * phi[i][qp]*phi[j][qp]
					 )
                                      );
                }
	      // The RHS contribution
	      Fe(i) += JxW[qp]*(
				// Time derivative (mass matrix) term
                                v_old*phi[i][qp]
				// Reaction term (explicit in scheme 0)
				- ( scheme==0 ?
				    dt*c_v2 * v_old*phi[i][qp]
				    : 0 )
				// Coupling term with u (live cell density)
				+ dt*c_v3 * u_old*phi[i][qp]
                                );

            }
        }

      // If this assembly program were to be used on an adaptive mesh,
      // we would have to apply any hanging node constraint equations
      dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

      // The element matrix and right-hand-side are now built
      // for this element.  Add them to the global matrix and
      // right-hand-side vector.  The SparseMatrix::add_matrix()
      // and NumericVector::add_vector() members do this for us.
      system.matrix->add_matrix (Ke, dof_indices);
      system.rhs->add_vector    (Fe, dof_indices);
    }

  // That concludes the system matrix assembly routine for v

#endif // #ifdef LIBMESH_ENABLE_AMR
}
