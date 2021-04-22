import numpy as np
from scipy.integrate import quad

from landlab import Component
from landlab.utils.return_array import return_array_at_node

from .cfuncs import calculate_qs_in,calculate_qs_in_lakeFiller

ROOT2 = np.sqrt(2.0)  # syntactic sugar for precalculated square root of 2
TIME_STEP_FACTOR = 0.5  # factor used in simple subdivision solver

from matplotlib.pyplot import pause
from ..depression_finder.lake_mapper import _FLOODED
import copy as cp

class Space_v2(Component):   

    _name = "Space_v2"

    _info = {
        "flow__link_to_receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "ID of link downstream of each node, which carries the discharge",
        },
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "flow__upstream_node_order": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array containing downstream-to-upstream ordered list of node IDs",
        },
        "sediment__flux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux (volume per unit time of sediment entering each node)",
        },
        "soil__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of soil or weathered bedrock",
        },
        "surface_water__discharge": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m**3/s",
            "mapping": "node",
            "doc": "Volumetric discharge of surface water",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
    }

    _cite_as = """@Article{gmd-10-4577-2017,
                  AUTHOR = {Shobe, C. M. and Tucker, G. E. and Barnhart, K. R.},
                  TITLE = {The SPACE~1.0 model: a~Landlab component for 2-D calculation of sediment transport, bedrock erosion, and landscape evolution},
                  JOURNAL = {Geoscientific Model Development},
                  VOLUME = {10},
                  YEAR = {2017},
                  NUMBER = {12},
                  PAGES = {4577--4604},
                  URL = {https://www.geosci-model-dev.net/10/4577/2017/},
                  DOI = {10.5194/gmd-10-4577-2017}
                  }"""

    def __init__(
        self,
        grid,
        K_sed=0.02,
        K_br=0.02,
        F_f=0.0,
        phi=0.3,
        H_star=0.1,
        v_s=1.0,
        v_s_lake=None,
        m_sp=0.5,
        n_sp=1.0,
        sp_crit_sed=0.0,
        sp_crit_br=0.0,
        discharge_field="surface_water__discharge",
        erode_flooded_nodes=False,
        thickness_lim= 100, 
        fillLakesToBrim = False,
    ):
        """Initialize the Space model.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        K_sed : float, field name, or array
            Erodibility for sediment (units vary).
        K_br : float, field name, or array
            Erodibility for bedrock (units vary).
        F_f : float
            Fraction of permanently suspendable fines in bedrock [-].
        phi : float
            Sediment porosity [-].
        H_star : float
            Sediment thickness required for full entrainment [L].
        v_s : float
            Effective settling velocity for chosen grain size metric [L/T].
        v_s_lake : float
            Effective settling velocity in lakes for chosen grain size metric [L/T].
        m_sp : float
            Drainage area exponent (units vary)
        n_sp : float
            Slope exponent (units vary)
        sp_crit_sed : float, field name, or array
            Critical stream power to erode sediment [E/(TL^2)]
        sp_crit_br : float, field name, or array
            Critical stream power to erode rock [E/(TL^2)]
        discharge_field : float, field name, or array
            Discharge [L^2/T]. The default is to use the grid field
            'surface_water__discharge', which is simply drainage area
            multiplied by the default rainfall rate (1 m/yr). To use custom
            spatially/temporally varying rainfall, use 'water__unit_flux_in'
            to specify water input to the FlowAccumulator.
        erode_flooded_nodes : bool (optional)
            Whether erosion occurs in flooded nodes identified by a
            depression/lake mapper (e.g., DepressionFinderAndRouter). When set
            to false, the field *flood_status_code* must be present on the grid
            (this is created by the DepressionFinderAndRouter). Default True.
            
        fillLakesToBrim : Wheter depostion should fill sinks to brim at water level {False}

        """
        if grid.at_node["flow__receiver_node"].size != grid.size("node"):
            msg = (
                "A route-to-multiple flow director has been "
                "run on this grid. The landlab development team has not "
                "verified that SPACE is compatible with "
                "route-to-multiple methods. Please open a GitHub Issue "
                "to start this process."
            )
            raise NotImplementedError(msg)
            
  
        super(Space_v2, self).__init__(
            grid,
            )
        
        self._soil__depth = grid.at_node["soil__depth"]
        self._topographic__elevation = grid.at_node["topographic__elevation"]

        if "bedrock__elevation" in grid.at_node:
            self._bedrock__elevation = grid.at_node["bedrock__elevation"]
        else:
            self._bedrock__elevation = grid.add_zeros(
                "bedrock__elevation", at="node", dtype=float
            )

            self._bedrock__elevation[:] = (
                self._topographic__elevation - self._soil__depth
            )


        # space specific inits
        self._thickness_lim=thickness_lim        
        self._H_star = H_star
        
        self._sed_erosion_term = np.zeros(grid.number_of_nodes)
        self._br_erosion_term = np.zeros(grid.number_of_nodes)
        self._Es = np.zeros(grid.number_of_nodes)
        self._Er = np.zeros(grid.number_of_nodes)

        # K's and critical values can be floats, grid fields, or arrays
        # use setters defined below
        self._K_sed = K_sed
        self._K_br = K_br

        self._sp_crit_sed = return_array_at_node(grid, sp_crit_sed)
        self._sp_crit_br = return_array_at_node(grid, sp_crit_br)
        
        self._erode_flooded_nodes = erode_flooded_nodes

        self._flow_receivers = grid.at_node["flow__receiver_node"]
        self._stack = grid.at_node["flow__upstream_node_order"]
        self._slope = grid.at_node["topographic__steepest_slope"]
        
        self.initialize_output_fields()

        self._qs = grid.at_node["sediment__flux"]
        self._q = return_array_at_node(grid, discharge_field)

        # Create arrays for sediment influx at each node, discharge to the
        # power "m", and deposition rate
        self._qs_in = np.zeros(grid.number_of_nodes)
        self._Q_to_the_m = np.zeros(grid.number_of_nodes)
        self._S_to_the_n = np.zeros(grid.number_of_nodes)
        # self._depo_rate = np.zeros(grid.number_of_nodes)

        # store other constants
        self._m_sp = np.float64(m_sp)
        self._n_sp = np.float64(n_sp)
        self._phi = np.float64(phi)
        self._v_s = np.float64(v_s)
        
        if v_s_lake ==None:
            self._v_s_lake = np.float64(v_s)
        else:
            self._v_s_lake = np.float64(v_s_lake)
        self._F_f = np.float64(F_f)
        # Boolean to 0 or 1 to be used as cython switch
        self._fillLakesToBrim=int(fillLakesToBrim)

        if phi >= 1.0:
            raise ValueError("Porosity must be < 1.0")

        if F_f > 1.0:
            raise ValueError("Fraction of fines must be <= 1.0")

        if phi < 0.0:
            raise ValueError("Porosity must be > 0.0")

        if F_f < 0.0:
            raise ValueError("Fraction of fines must be > 0.0")
            
        if not isinstance(fillLakesToBrim, bool):
            raise ValueError("fillLakesToBrim must be True or False")
            
        # If filling to brim, a depression free field must be provided. 
        # 'deprFree_elevation' is automatically produced as a grid field when using FlowAccumulatorPf
        if fillLakesToBrim:
            if not 'deprFree_elevation' in self.grid.at_node.keys():
                raise NotImplementedError("If filling to brim, a depression free \
                                          field must be provided. \
                                          'deprFree_elevation' is automatically \
                                              produced as a grid field when using \
                                                  FlowAccumulatorPf")
    @property
    def K_br(self):
        """Erodibility of bedrock(units depend on m_sp)."""
        return self._K_br

    @K_br.setter
    def K_br(self, new_val):
        self._K_br = return_array_at_node(self._grid, new_val)

    @property
    def K_sed(self):
        """Erodibility of sediment(units depend on m_sp)."""
        return self._K_sed

    @K_sed.setter
    def K_sed(self, new_val):
        self._K_sed = return_array_at_node(self._grid, new_val)
        
    @property
    def Es(self):
        """Sediment erosion term."""
        return self._Es

    @property
    def Er(self):
        """Bedrock erosion term."""
        return self._Er

    @property
    def H(self):
        """Sediment thickness."""
        return self._H            
        



     

    def _calc_erosion_rates(self):
        """Calculate erosion rates."""       
        
        """Calculate erosion rates."""
        
        br = self.grid.at_node["bedrock__elevation"]
        H = self.grid.at_node["soil__depth"]
        
        # if sp_crits are zero, then this colapses to correct all the time.
        if self._n_sp == 1.0:
            S_to_the_n = self._slope
        else:
            S_to_the_n = np.power(self._slope, self._n_sp)
        omega_sed = self._K_sed * self._Q_to_the_m * S_to_the_n
        omega_br = self._K_br * self._Q_to_the_m * S_to_the_n

        omega_sed_over_sp_crit = np.divide(
            omega_sed,
            self._sp_crit_sed,
            out=np.zeros_like(omega_sed),
            where=self._sp_crit_sed != 0,
        )

        omega_br_over_sp_crit = np.divide(
            omega_br,
            self._sp_crit_br,
            out=np.zeros_like(omega_br),
            where=self._sp_crit_br != 0,
        )

        self._sed_erosion_term = omega_sed - self._sp_crit_sed * (
            1.0 - np.exp(-omega_sed_over_sp_crit)
        )/ (
            1 - self._phi
        )  # convert from a volume to a mass flux.
        self._br_erosion_term = omega_br - self._sp_crit_br * (
            1.0 - np.exp(-omega_br_over_sp_crit)
        )
        
        
        # Space does not allow the formation of potholes (addition v2)
        r = self._grid.at_node["flow__receiver_node"]
        br_e_max= br - br[r]
        br_e_max[br_e_max<0]=0
        self._br_erosion_term= np.minimum(self._br_erosion_term,br_e_max)     
        
        self._Es = self._sed_erosion_term * (
            1.0 - np.exp(-H / self._H_star)
        )
        self._Er = self._br_erosion_term * np.exp(-H/self._H_star)
        
        ''' if the soil layer becomes exceptionally thick (e.g. because of 
        landslide derived sediment deposition(,) the algorithm will become 
        unstable because np.exp(x) with x > 709 yeilds inf values. 
        Therefore soil depth is temporqlly topped of at 200m and the remaining 
        values are added back after the space component has run '''
        
        self._Es[H > self._thickness_lim] = self._sed_erosion_term[H > self._thickness_lim] 
        self._Er[H > self._thickness_lim] = 0        
    

   
    def run_one_step_basic(self, dt = 10):        
        
        
        z = self.grid.at_node["topographic__elevation"]
        br = self.grid.at_node["bedrock__elevation"]
        H = self.grid.at_node["soil__depth"]
        area = self.grid.cell_area_at_node
        
        r = self.grid.at_node["flow__receiver_node"]
        stack = self.grid.at_node["flow__upstream_node_order"]        
        slope = self.grid.at_node["topographic__steepest_slope"]     
        
        # Choose a method for calculating erosion:
        self._Q_to_the_m[:] = np.power(self._q, self._m_sp)
        self._calc_erosion_rates()  
        
        if "flood_status_code" in self.grid.at_node.keys():
            flood_status = self.grid.at_node["flood_status_code"]
            flooded_nodes = np.nonzero(flood_status == _FLOODED)[0]                
        else: 
            flooded_nodes = np.nonzero([slope<0])[1]                
        
        self._Es[flooded_nodes] = 0.0
        self._Er[flooded_nodes] = 0.0
        self._sed_erosion_term[flooded_nodes] = 0.0
        self._br_erosion_term[flooded_nodes] = 0.0       


        self._qs_in[:] = 0     
        
        if self._fillLakesToBrim or self._v_s!=self._v_s_lake:   
            
            v = np.ones(self.grid.number_of_nodes)*self._v_s
            v[flooded_nodes] = self._v_s_lake
            if self._fillLakesToBrim:            
                waterEl = self.grid.at_node["deprFree_elevation"]
            else:
                waterEl = np.zeros(self.grid.number_of_nodes)     

            vol_SSY_riv = calculate_qs_in_lakeFiller(
                np.flipud(stack),
                r,
                area,
                self._q,
                self._qs,
                self._qs_in,
                self._Es,
                self._Er,
                self._Q_to_the_m,
                slope ,
                v,
                H,
                br,
                waterEl,
                self._sed_erosion_term,
                self._br_erosion_term,
                self._phi,
                self._F_f,
                self._K_sed,
                self._H_star,
                dt,                
                self._thickness_lim,
                self._fillLakesToBrim,
            )
            
        
        else:
            vol_SSY_riv = calculate_qs_in(
                np.flipud(stack),
                r,
                area,
                self._q,
                self._qs,
                self._qs_in,
                self._Es,
                self._Er,
                self._Q_to_the_m,
                slope ,
                H,
                br,
                self._sed_erosion_term,
                self._br_erosion_term,
                self._v_s,
                self._phi,
                self._F_f,
                self._K_sed,
                self._H_star,
                dt,
                self._thickness_lim
            )


        V_leaving_riv = np.sum(self._qs_in)*dt    
        # Update topography
        cores = self._grid.core_nodes
        z[cores] = (br[cores] + H[cores])  
        
        return vol_SSY_riv, V_leaving_riv 
        
    
    
    def run_one_step(self, dt):
        vol_SSY_riv, V_leaving_riv = self.run_one_step_basic(dt)
        return vol_SSY_riv, V_leaving_riv 
    