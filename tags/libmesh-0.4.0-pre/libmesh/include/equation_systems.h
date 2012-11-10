// $Id: equation_systems.h,v 1.23 2003-05-04 23:58:51 benkirk Exp $

// The Next Great Finite Element Library.
// Copyright (C) 2002  Benjamin S. Kirk, John W. Peterson
  
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



#ifndef __equation_systems_h__
#define __equation_systems_h__

// C++ includes

// Local Includes
#include "xdr_cxx.h"
#include "equation_systems_base.h"


// Forward Declarations


/**
 * This contains one or more equation systems that are
 * to be solved in a simulation.  These equation systems
 * are identified by a user-specified name and are solved
 * in the order that they are declared.
 *
 * @author Benjamin S. Kirk, 2002-2003
 */

// ------------------------------------------------------------
// EquationSystems<T_sys> class definition
template <typename T_sys>
class EquationSystems : public EquationSystemsBase
{
public:

  /**
   * Constructor.  Optionally initializes required
   * data structures. 
   */
  EquationSystems (Mesh& mesh);

  /**
   * Destructor.
   */
  ~EquationSystems ();
 
  /**
   * Returns tha data structure to a pristine state.
   */
  void clear ();
  
  /**
   * Initialize all the systems
   */
  void init ();

  /**
   * Reinitialize all the systems
   */
  void reinit ();

  /**
   * @returns the number of equation systems.
   */
  unsigned int n_systems() const;

  /**
   * Add the system named \p name to the systems array.
   */
  void add_system (const std::string& name);
  
  /**
   * Remove the system named \p name from the systems array.
   */
  void delete_system (const std::string& name);

  /**
   * @returns the total number of variables in all
   * systems.
   */
  unsigned int n_vars () const;
  
  /**
   * @returns the total number of degrees of freedom
   * in all systems.
   */
  unsigned int n_dofs () const;

  /**
   * @returns a reference to the system named \p name.
   */
  T_sys & operator () (const std::string& name);

  /**
   * @returns a constant reference to the system name
   */
  const T_sys & operator () (const std::string& name) const;

  /**
   * @returns a reference to system number \p num.
   */
  T_sys & operator () (const unsigned int num);

  /**
   * @returns a constant reference to system number \p num.
   */
  const T_sys & operator () (const unsigned int num) const;

  /**
   * @returns the name of the system number num.
   */
  const std::string & name (const unsigned int num) const;
  
  /**
   * Fill the input vector \p var_names with the names
   * of the variables for each system.
   */
  void build_variable_names (std::vector<std::string>& var_names);

  /**
   * Fill the input vector \p soln with the solution values for the
   * system named \p name.  Note that the input
   * vector \p soln will only be assembled on processor 0, so this
   * method is only applicable to outputting plot files from processor 0.
   */
  void build_solution_vector (std::vector<Number>& soln,
			      std::string& system_name,
			      std::string& variable_name);
  
  /**
   * Fill the input vector \p soln with solution values.  The
   * entries will be in variable-major format (corresponding to
   * the names from \p build_variable_names()).  Note that the input
   * vector \p soln will only be assembled on processor 0, so this
   * method is only applicable to outputting plot files from processor 0.
   */
  void build_solution_vector (std::vector<Number>& soln);
  
  /**
   * Read & initialize the systems from disk using the XDR data format.
   * This format allows for machine-independent binary output.
   *
   * Note that the equation system can be defined without initializing
   * the data vectors to any solution values.  This can be done
   * by calling the routine with the read_data flag set to false.
   */
  void read(const std::string& name,
	    const Xdr::XdrMODE,
	    const bool read_header=true,
	    const bool read_data=true,
	    const bool read_additional_data=true);

  /**
   * Write the systems to disk using the XDR data format.
   * This format allows for machine-independent binary output.
   *
   * Note that the solution data can be omitted by calling
   * this routine with the write_data flag set to false.
   */
  void write(const std::string& name,
	     const Xdr::XdrMODE,
	     const bool write_data=true,
	     const bool write_additional_data=true);

  /**
   * @returns \p true when this equation system contains
   * identical data, up to the given threshold.  Delegates
   * most of the comparisons to perform to the responsible
   * systems
   */
  bool compare (const EquationSystems<T_sys>& other_es, 
		const Real threshold,
		const bool verbose) const;

  /**
   * Prints information about the equation systems.
   */
  void print_info () const;

  /**
   * @returns a string containing information about the
   * equation systems.
   */
  std::string get_info() const;
  
  
 protected:
  
  /**
   * Data structure that holds the systems.
   */
  std::map<std::string, T_sys*> _systems;

};



// ------------------------------------------------------------
// EquationSystems inline methods
template <class T_sys>
inline
unsigned int EquationSystems<T_sys>::n_systems () const
{
  return _systems.size();
}



template <class T_sys>
inline
void EquationSystems<T_sys>::print_info() const
{
  std::cout << this->get_info() 
	    << EquationSystemsBase::get_info()
	    << std::endl;
}

#endif