"""
Created on Wed May 31 11:41:20 2023

@author: LaurentRoberge
"""

import numpy as np

from landlab import Component, LinkStatus
from landlab.grid.mappers import map_value_at_max_node_to_link
from landlab.utils.return_array import return_array_at_node


class ConcentrationTrackerProductionDecay(Component):

    """This component allows production and decay for the concentration of any
    user-defined property of sediment. The user may define a production rate
    and a decay rate for the property concentration in surface sediments and in
    bedrock. These rates can be individual values, arrays, timeseries, or user-
    defined functions dependent on Landlab grid fields or other variables.
    Production rates and decay rates must both be positive (or in the
    case of functions, must generate positive values), as the component adds 
    and subtracts the values as necessary:
        - a positive production rate will cause an increase in concentration.
        - a positive decay rate will cause a decrease in concentration.
    
    This component should be coupled with one or more ConcentrationTracker
    components, which track the property concentration in sediment and in 
    bedrock across a Landlab grid. The ConcentrationTracker components use a 
    mass balance approach in which production and/or decay of the sediment 
    property can be properly applied each timestep using this production/decay 
    component.
    
    Examples
    --------
    A 1-D hillslope with no sediment movement and an integer production rate:

    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components import ConcentrationTrackerProductionDecay
    >>> mg = RasterModelGrid((3, 5),xy_spacing=2.)
    >>> mg.set_status_at_node_on_edges(right=4, top=4, left=4, bottom=4)
    >>> mg.status_at_node[5] = mg.BC_NODE_IS_FIXED_VALUE
    >>> c = mg.add_zeros('sediment_property__concentration', at='node')
    >>> h = mg.add_zeros("soil__depth", at="node")
    >>> z_br = mg.add_zeros("bedrock__elevation", at="node")
    >>> z = mg.add_zeros("topographic__elevation", at="node")
    >>> _ = mg.add_zeros('soil_production__rate', at='node')
    ### ADD EXAMPLE BELOW ###
    >>> c[8] += 1
    >>> h += mg.node_x
    >>> z_br += mg.node_x
    >>> z += z_br + h
    >>> ddd = DepthDependentDiffuser(mg)
    >>> ct = ConcentrationTrackerForDiffusion(mg)
    >>> ddd.run_one_step(1.)
    >>> ct.run_one_step(1.)
    >>> np.allclose(mg.at_node["topographic__elevation"][mg.core_nodes],
    ...             np.array([4.11701964, 8.01583689, 11.00247875]))
    True
    >>> np.allclose(mg.at_node["sediment_property__concentration"][mg.core_nodes],
    ...             np.array([0., 0.24839685, 1.]))
    True
    ### ADD EXAMPLE ABOVE ###
    
    Now, the same hillslope with a user-defined decay rate function depending
    only on the pre-existing concentration value (e.g., half-life):
    ### ADD EXAMPLE BELOW ###

    ### SHOW EXAMPLE OF HALF-LIFE DECAY FUNCTION ###

    ### ADD EXAMPLE BELOW ###
        
    Now, changing the production rate to a user-defined function that depends
    on topographic elevation (which needs to be added as an input):
    ### ADD EXAMPLE BELOW ###

    ### SHOW EXAMPLE WITH TOPO DEPENDENCE ###

    ### ADD EXAMPLE BELOW ###
    
    Finally, the same hillslope with the production and decay functions, now 
    including downslope soil transport using the DepthDependentDiffuser:
    ### ADD EXAMPLE BELOW ###
    >>> from landlab.components import DepthDependentDiffuser
    >>> from landlab.components import ConcentrationTrackerForDiffusion
    
    ### ADD DEPTHDEPENDENTDIFFUSER AND CONCENTRATIONTRACKERFORDIFFUSION ###

    ### ADD EXAMPLE BELOW ###
    
    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    CITATION

    """

    _name = "ConcentrationTrackerProductionDecay"

    _unit_agnostic = True

    _cite_as = """
    CITATION
    """

    _info = {
        "sediment_property__concentration": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-/m^3",
            "mapping": "node",
            "doc": "Mass concentration of property per unit volume of sediment",
        },
        "bedrock_property__concentration": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-/m^3",
            "mapping": "node",
            "doc": "Mass concentration of property per unit volume of bedrock",
        },
        "sediment_property_production__rate": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-/m^3/yr",
            "mapping": "node",
            "doc": "Production rate of property per unit volume of sediment per time",
        },
        "sediment_property_decay__rate": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-/m^3/yr",
            "mapping": "node",
            "doc": "Decay rate of property per unit volume of sediment per time",
        },
        "bedrock_property_production__rate": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-/m^3/yr",
            "mapping": "node",
            "doc": "Production rate of property per unit volume of bedrock per time",
        },
        "bedrock_property_decay__rate": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-/m^3/yr",
            "mapping": "node",
            "doc": "Decay rate of property per unit volume of bedrock per time",
        },
    }

    def __init__(
        self,
        grid,
        concentration_initial=0,
        concentration_in_bedrock=0,
        local_production_rate_in_sediment=0,
        local_decay_rate_in_sediment=0,
        local_production_rate_in_bedrock=0,
        local_decay_rate_in_bedrock=0,
    ):
        """
        Parameters
        ----------
        grid: ModelGrid
            Landlab ModelGrid object
        concentration_initial: positive float, array, or field name (optional)
            Initial concentration in soil/sediment, -/m^3
        concentration_in_bedrock: positive float, array, or field name (optional)
            Concentration in bedrock, -/m^3
        local_production_rate_in_sediment: float, array, or field name (optional)
            Rate of local production in sediment, -/m^3/yr
        local_decay_rate_in_sediment: float, array, or field name (optional)
            Rate of local decay in sediment, -/m^3/yr
        local_production_rate_in_bedrock: float, array, or field name (optional)
            Rate of local production in bedrock, -/m^3/yr
        local_decay_rate_in_bedrock: float, array, or field name (optional)
            Rate of local decay in bedrock, -/m^3/yr
        """

        super().__init__(grid)
        # Store grid and parameters

        # use setters for C_init, C_br, P_sed, D_sed, P_br, and D_br defined below
        self.C_init = concentration_initial
        self.C_br = concentration_in_bedrock
        self.P_sed = local_production_rate_in_sediment
        self.D_sed = local_decay_rate_in_sediment
        self.P_br = local_production_rate_in_bedrock
        self.D_br = local_decay_rate_in_bedrock
        
        # Arrays for production and decay per timestep
        self._P_sed_in_timestep = np.zeros(self._grid.number_of_nodes)
        self._D_sed_in_timestep = np.zeros(self._grid.number_of_nodes)
        self._P_br_in_timestep = np.zeros(self._grid.number_of_nodes)
        self._D_br_in_timestep = np.zeros(self._grid.number_of_nodes)

        # create outputs if necessary and get reference.
        self.initialize_output_fields()

        # Define concentration field (if all zeros, then add C_init)
        if not self._grid.at_node["sediment_property__concentration"].any():
            self._grid.at_node["sediment_property__concentration"] += self.C_init
        self._concentration = self._grid.at_node["sediment_property__concentration"]

        if not self._grid.at_node["bedrock_property__concentration"].any():
            self._grid.at_node["bedrock_property__concentration"] += self.C_br
        self.C_br = self._grid.at_node["bedrock_property__concentration"]

        if not self._grid.at_node["sediment_property_production__rate"].any():
            self._grid.at_node["sediment_property_production__rate"] += self.P_sed
        self.P_sed = self._grid.at_node["sediment_property_production__rate"]

        if not self._grid.at_node["sediment_property_decay__rate"].any():
            self._grid.at_node["sediment_property_decay__rate"] += self.D_sed
        self.D_sed = self._grid.at_node["sediment_property_decay__rate"]
        
        if not self._grid.at_node["bedrock_property_production__rate"].any():
            self._grid.at_node["bedrock_property_production__rate"] += self.P_br
        self.P_br = self._grid.at_node["bedrock_property_production__rate"]

        if not self._grid.at_node["bedrock_property_decay__rate"].any():
            self._grid.at_node["bedrock_property_decay__rate"] += self.D_br
        self.D_br = self._grid.at_node["bedrock_property_decay__rate"]

        # Check that concentration values are within physical limits
        if isinstance(concentration_initial, np.ndarray):
            if concentration_initial.any() < 0:
                raise ValueError("Concentration cannot be negative.")
        else:
            if concentration_initial < 0:
                raise ValueError("Concentration cannot be negative.")

        if isinstance(concentration_in_bedrock, np.ndarray):
            if concentration_in_bedrock.any() < 0:
                raise ValueError("Concentration in bedrock cannot be negative.")
        else:
            if concentration_in_bedrock < 0:
                raise ValueError("Concentration in bedrock cannot be negative.")

    @property
    def C_init(self):
        """Initial concentration in soil/sediment (kg/m^3)."""
        return self._C_init

    @property
    def C_br(self):
        """Concentration in bedrock (kg/m^3)."""
        return self._C_br

    @property
    def P_sed(self):
        """Rate of local production in sediment (kg/m^3/yr)."""
        return self._P_sed

    @property
    def D_sed(self):
        """Rate of local decay in sediment (kg/m^3/yr)."""
        return self._D_sed
    
    @property
    def P_br(self):
        """Rate of local production in bedrock (kg/m^3/yr)."""
        return self._P_br

    @property
    def D_br(self):
        """Rate of local decay in bedrock (kg/m^3/yr)."""
        return self._D_br

    @C_init.setter
    def C_init(self, new_val):
        self._C_init = return_array_at_node(self._grid, new_val)

    @C_br.setter
    def C_br(self, new_val):
        self._C_br = return_array_at_node(self._grid, new_val)

    @P_sed.setter
    def P_sed(self, new_val):
        self._P_sed = return_array_at_node(self._grid, new_val)

    @D_sed.setter
    def D_sed(self, new_val):
        self._D_sed = return_array_at_node(self._grid, new_val)
    
    @P_br.setter
    def P_br(self, new_val):
        self._P_br = return_array_at_node(self._grid, new_val)

    @D_br.setter
    def D_br(self, new_val):
        self._D_br = return_array_at_node(self._grid, new_val)

    def production_in_sediment(self, dt):
        """Calculate change in concentration due to production in sediment for
        the time period 'dt'.

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """
        
        # Zero out array
        self._P_sed_in_timestep[:] = 0
                        
        ################## SORT OUT EQUATION BELOW
        
        # Calculate production
        with np.errstate(divide="ignore", invalid="ignore"):
            self._P_sed_in_timestep = (dt * self._P_sed / 2) * (
                                self._soil__depth_old / self._soil__depth + 1
            )
        
        # Replace nan values (from dividing by zero soil depth)
        np.nan_to_num(self._P_sed_in_timestep, copy=False)
                
        ################## SORT OUT EQUATION ABOVE
        
        
    def decay_in_sediment(self, dt):
        """Calculate change in concentration due to decay in sediment for the
        time period 'dt'.

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """
        
        # Zero out array
        self._D_sed_in_timestep[:] = 0
                        
        ################## SORT OUT EQUATION BELOW
        
        # Calculate decay
        with np.errstate(divide="ignore", invalid="ignore"):
            self._D_sed_in_timestep = (dt * self._D_sed / 2) * (
                                self._soil__depth_old / self._soil__depth + 1
                                )
        
        # Replace nan values (from dividing by zero soil depth)
        np.nan_to_num(self._D_sed_in_timestep, copy=False)
                        
        ################## SORT OUT EQUATION ABOVE
        
   
    def production_in_bedrock(self, dt):
        """Calculate change in concentration due to production in bedrock for
        the time period 'dt'.

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """
        
        # Zero out array
        self._P_br_in_timestep[:] = 0
        
        ################## SORT OUT EQUATION BELOW
        
        # Calculate production
        with np.errstate(divide="ignore", invalid="ignore"):
            self._P_br_in_timestep = (dt * self._P_br / 2) * (
                               self._soil__depth_old / self._soil__depth + 1
            )
        
        # Replace nan values (from dividing by zero soil depth)
        np.nan_to_num(self._P_br_in_timestep, copy=False)
        
        ################## SORT OUT EQUATION ABOVE
            
        
    def decay_in_bedrock(self, dt):
        """Calculate change in concentration due to decay in bedrock for the
        time period 'dt'.

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """

        # Zero out array
        self._D_br_in_timestep[:] = 0
                
        ################## SORT OUT EQUATION BELOW
        
        # Calculate decay
        with np.errstate(divide="ignore", invalid="ignore"):
            self._D_br_in_timestep = (dt * self._D_br / 2) * (
                               self._soil__depth_old / self._soil__depth + 1
                               )
        
        # Replace nan values (from dividing by zero soil depth)
        np.nan_to_num(self._D_br_in_timestep, copy=False)
                
        ################## SORT OUT EQUATION ABOVE
        

    def calculate_concentration(self, dt):
        """Calculate new concentration values due to production and decay for
        the time period 'dt'.

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """
        
        # Calculate concentration
        self._concentration[:] += self._P_sed_in_timestep
        self._concentration[:] -= self._D_sed_in_timestep
        self.C_br[:] += self._P_br_in_timestep
        self.C_br[:] -= self._D_br_in_timestep

        # Replace nan values (from dividing by zero soil depth)
        np.nan_to_num(self._concentration, copy=False)

    def run_one_step(self, dt):
        """

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """

        self.production_in_sediment(dt)
        self.decay_in_sediment(dt)
        self.production_in_bedrock(dt)
        self.decay_in_bedrock(dt)
        self.calculate_concentration(dt)
