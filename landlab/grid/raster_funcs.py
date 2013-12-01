import numpy as np

from .base import INACTIVE_BOUNDARY


def _make_optional_arg_into_array(number_of_elements, *args):
    assert(len(args) < 2)
    if len(args) == 0:
        ids = np.arange(number_of_elements)
    else:
        ids = args[0]
        if not isinstance(ids, list) or not isinstance(ids, np.ndarray):
            try:
                ids = list(ids)
            except TypeError:
                ids = [ids]
    return ids


def calculate_gradient_across_cell_faces(grid, node_values, *args, **kwds):
    """calculate_gradient_across_cell_faces(grid, node_values, [cell_ids], out=None)
    """
    cell_ids = _make_optional_arg_into_array(grid.number_of_cells, *args)
    node_ids = grid.node_index_at_cells[cell_ids]

    values_at_neighbors = node_values[grid.get_neighbor_list(node_ids)]
    values_at_nodes = node_values[node_ids].reshape(len(node_ids), 1)

    out = np.subtract(values_at_nodes, values_at_neighbors, **kwds)
    out *= 1. / grid.node_spacing

    return out


def calculate_gradient_across_cell_corners(grid, node_values, *args, **kwds):
    """calculate_gradient_across_cell_corners(grid, node_values, [cell_ids], out=None)
    """
    cell_ids = _make_optional_arg_into_array(grid.number_of_cells, *args)
    node_ids = grid.node_index_at_cells[cell_ids]

    values_at_diagonals = node_values[grid.get_diagonal_list(node_ids)]
    values_at_nodes = node_values[node_ids].reshape(len(node_ids), 1)

    out = np.subtract(values_at_nodes, values_at_diagonals, **kwds)
    np.divide(out, np.sqrt(2.) * grid.node_spacing, out=out)

    return out


def calculate_max_gradient_across_adjacent_cells(grid, node_values, *args, **kwds):
    """calculate_max_gradient_across_adjacent_cells(grid, node_values, [cell_ids], method='d4', out=None)
    """
    method = kwds.pop('method', 'd4')

    if method.lower() == 'd4':
        return calculate_max_gradient_across_cell_faces(
            grid, node_values, *args, **kwds)
    elif method.lower() == 'd8':
        neighbor_grads = calculate_max_gradient_across_cell_faces(
            grid, node_values, *args, **kwds)
        diagonal_grads = calculate_max_gradient_across_cell_corners(
            grid, node_values, *args, **kwds)

        return_face = kwds.pop('return_face', False)

        if not return_face:
            return np.choose(neighbor_grads > diagonal_grads,
                             (diagonal_grads, neighbor_grads), **kwds)
        else:
            max_grads = np.choose(neighbor_grads[0] > diagonal_grads[0],
                                  (diagonal_grads[0], neighbor_grads[0]),
                                  **kwds)
            inds = np.choose(neighbor_grads[0] > diagonal_grads[0],
                                  (diagonal_grads[1] + 4, neighbor_grads[1]),
                                  **kwds)
            return (max_grads, inds)


def calculate_max_gradient_across_cell_corners(grid, node_values, *args,
                                               **kwds):
    """calculate_max_gradient_across_cell_corners(grid, node_values [, cell_ids], return_face=False, out=None)
    """
    return_face = kwds.pop('return_face', False)

    cell_ids = _make_optional_arg_into_array(grid.number_of_cells, *args)

    grads = calculate_gradient_across_cell_corners(grid, node_values, cell_ids)

    if return_face:
        ind = np.argmax(grads, axis=1)
        if 'out' not in kwds:
            out = np.empty(len(cell_ids), dtype=grads.dtype)
        out[:] = grads[xrange(len(cell_ids)), ind]
        return (out, 3 - ind)
    else:
        return grads.max(axis=1, **kwds)


def calculate_max_gradient_across_cell_faces(grid, node_values, *args, **kwds):
    """calculate_max_gradient_across_cell_faces(grid, node_values, [cell_ids], return_face=False, out=None)

    This method calculates the gradients in *node_values* across all four
    faces of the cell or cells with ID *cell_ids*. Slopes downward from the
    cell are reported as positive. If *cell_ids* is not given, calculate
    gradients for all cells.

    Use the *return_face* keyword to return a tuple, with the first element
    being the gradients and the second the *cell-level* id of the face with
    the max gradient. This is the id of face as measured within the cell and
    so ranges from 0 to 3 for a cell with four sides.

    >>> from landlab import RasterModelGrid
    >>> rmg = RasterModelGrid(3, 3)
    >>> values_at_nodes = np.arange(9.)
    >>> calculate_max_gradient_across_cell_faces(rmg, values_at_nodes)
    array([ 3.])
    >>> (_, ind) = calculate_max_gradient_across_cell_faces(rmg, values_at_nodes, return_face=True)
    >>> ind
    array([0])
    """
    return_face = kwds.pop('return_face', False)

    cell_ids = _make_optional_arg_into_array(grid.number_of_cells, *args)

    grads = calculate_gradient_across_cell_faces(grid, node_values, cell_ids)

    if return_face:
        ind = np.argmax(grads, axis=1)
        if 'out' not in kwds:
            out = np.empty(len(cell_ids), dtype=grads.dtype)
        out[:] = grads[xrange(len(cell_ids)), ind]
        return (out, 3 - ind)
    else:
        return grads.max(axis=1, **kwds)


def active_link_id_of_cell_neighbor(grid, inds, *args):
    """ active_link_id_of_cell_neighbor(grid, link_ids [, cell_ids])

    Return an array of the active link ids for neighbors of *cell_id* cells.
    *link_ids* is an index into the links of a cell as measured
    clockwise starting from the south.

    If *cell_ids* is not given, return neighbors for all cells in the grid.
    """
    cell_ids = _make_optional_arg_into_array(grid.number_of_cells, *args)
    node_ids = grid.node_index_at_cells[cell_ids]
    links = grid.active_node_links(node_ids).T

    if not isinstance(inds, np.ndarray):
        inds = np.array(inds)

    return links[xrange(len(cell_ids)), inds]


def node_id_of_cell_neighbor(grid, inds, *args):
    """ node_id_of_cell_neighbor(grid, neighbor_ids [, cell_ids])

    Return an array of the node ids for neighbors of *cell_id* cells.
    *neighbor_ids* is an index into the neighbors of a cell as measured
    clockwise starting from the south.

    If *cell_ids* is not given, return neighbors for all cells in the grid.
    """
    cell_ids = _make_optional_arg_into_array(grid.number_of_cells, *args)
    node_ids = grid.node_index_at_cells[cell_ids]
    neighbors = grid.get_neighbor_list(node_ids)

    if not isinstance(inds, np.ndarray):
        inds = np.array(inds)

    return neighbors[xrange(len(cell_ids)), 3 - inds]


def node_id_of_cell_corner(grid, inds, *args):
    """ node_id_of_cell_corner(grid, corner_ids [, cell_ids])

    Return an array of the node ids for diagonal neighbors of *cell_id* cells.
    *corner_ids* is an index into the corners of a cell as measured
    clockwise starting from the southeast.

    If *cell_ids* is not given, return neighbors for all cells in the grid.
    """
    cell_ids = _make_optional_arg_into_array(grid.number_of_cells, *args)
    node_ids = grid.node_index_at_cells[cell_ids]
    diagonals = grid.get_diagonal_list(node_ids)

    if not isinstance(inds, np.ndarray):
        inds = np.array(inds)

    return diagonals[xrange(len(cell_ids)), 3 - inds]


def calculate_flux_divergence_at_nodes(grid, active_link_flux, out=None):
    """
    Same as calculate_flux_divergence_at_active_cells, but works with and
    returns a list of net unit fluxes that corresponds to all nodes, rather
    than just active cells.
    
    Note that we DO compute net unit fluxes at boundary nodes (even though
    these don't have active cells associated with them, and often don't have 
    cells of any kind, because they are on the perimeter). It's up to the 
    user to decide what to do with these boundary values.
    
    Example:

    >>> from landlab import RasterModelGrid
    >>> rmg = RasterModelGrid(4, 5, 1.0)
    >>> u = [0., 1., 2., 3., 0.,
    ...      1., 2., 3., 2., 3.,
    ...      0., 1., 2., 1., 2.,
    ...      0., 0., 2., 2., 0.]
    >>> u = np.array(u)
    >>> grad = rmg.calculate_gradients_at_active_links(u)
    >>> grad
    array([ 1.,  1., -1., -1., -1., -1., -1.,  0.,  1.,  1.,  1., -1.,  1.,
            1.,  1., -1.,  1.])
    >>> flux = -grad    # downhill flux proportional to gradient
    >>> df = rmg.calculate_flux_divergence_at_nodes(flux)
    >>> df
    array([ 0., -1., -1.,  1.,  0., -1.,  2.,  4., -2.,  1., -1.,  0.,  1.,
           -4.,  1.,  0., -1.,  0.,  1.,  0.])
        
    If calculate_gradients_at_nodes is called inside a loop, you can
    improve speed by creating an array outside the loop. For example, do
    this once, before the loop:
        
    >>> df = rmg.zeros(centering='node') # outside loop
    >>> rmg.number_of_nodes
    20
        
    Then do this inside the loop:
        
    >>> df = rmg.calculate_flux_divergence_at_nodes(flux, df)
        
    In this case, the function will not have to create the df array.
    """
    assert (len(active_link_flux) == grid.number_of_active_links), \
           "incorrect length of active_link_flux array"
        
    # If needed, create net_unit_flux array
    if out is None:
        out = grid.empty(centering='node')
    out.fill(0.)
    net_unit_flux = out
        
    assert(len(net_unit_flux) == grid.number_of_nodes)
    
    flux = np.zeros(len(active_link_flux) + 1)
    flux[:len(active_link_flux)] = active_link_flux * grid._dx
    
    net_unit_flux[:] = (
        (flux[grid.node_active_outlink_matrix[0][:]] +
         flux[grid.node_active_outlink_matrix[1][:]]) -
        (flux[grid.node_active_inlink_matrix[0][:]] +
         flux[grid.node_active_inlink_matrix[1][:]])) / grid.cellarea

    return net_unit_flux


# TODO: Functions below here still need to be refactored for speed and to
# conform to the interface standards.

def calculate_max_gradient_across_node(grid, u, cell_id):
    """
    This method calculates the gradients in u across all 4 faces of the 
    cell with ID cell_id, and across the four diagonals. It then returns 
    the steepest (most negative) of these values, followed by its dip 
    direction (e.g.: 0.12, 225). i.e., this is a D8 algorithm. Slopes 
    downward from the cell are reported as positive.
        
    This code is actually calculating slopes, not gradients.  
    The max gradient is the most negative, but the max slope is the most
    positive.  So, this was updated to return the max value, not the 
    min.
        
    GT: Might be possible to speed this up using inlink_matrix and 
    outlink_matrix.
    """
    #We have poor functionality if these are edge cells! Needs an exception
    neighbor_cells = grid.get_neighbor_list(cell_id)
    neighbor_cells.sort()        
    diagonal_cells = []
    if neighbor_cells[0]!=-1:
        diagonal_cells.extend([neighbor_cells[0]-1, neighbor_cells[0]+1])
    if neighbor_cells[3]!=-1:
        diagonal_cells.extend([neighbor_cells[3]-1, neighbor_cells[3]+1])
    slopes = []
    diagonal_dx = np.sqrt(2.) * grid._dx  # Corrected (multiplied grid._dx) SN 05Nov13
    for a in neighbor_cells:
        #ng I think this is actually slope as defined by a geomorphologist,
        #that is -dz/dx and not the gradient (dz/dx)
        #print '\n', cell_id
        #print '\n', a
        single_slope = (u[cell_id] - u[a])/grid._dx
        #print 'cell id: ', cell_id
        #print 'neighbor id: ', a
        #print 'cell, neighbor are internal: ', grid.is_interior(cell_id), grid.is_interior(a)
        #print 'cell elev: ', u[cell_id]
        #print 'neighbor elev: ', u[a]
        #print single_slope
        if not np.isnan(single_slope): #This should no longer be necessary, but retained in case
            slopes.append(single_slope)
        else:
            print 'NaNs present in the grid!'
            
    for a in diagonal_cells:
        single_slope = (u[cell_id] - u[a])/(diagonal_dx)
        #print single_slope
        if not np.isnan(single_slope):
            slopes.append(single_slope)
        else:
            print 'NaNs present in the grid!'
    #print 'Slopes list: ', slopes
    #ng thinks that the maximum slope should be found here, not the 
    #minimum slope, old code commented out.  New code below it.
    #if slopes:
    #    min_slope, index_min = min((min_slope, index_min) for (index_min, min_slope) in enumerate(slopes))
    #else:
    #    print u
    #    print 'Returning NaN angle and direction...'
    #    min_slope = np.nan
    #    index_min = 8
    if slopes:
        max_slope, index_max = max((max_slope, index_max) for (index_max, max_slope) in enumerate(slopes))
    else:
        print u
        print 'Returning NaN angle and direction...'
        max_slope = np.nan
        index_max = 8
    
    # North = Zero Radians  = Clockwise Positive
    angles = [180., 270., 90., 0., 225., 135., 315., 45., np.nan] #This is inefficient 
    
    #ng commented out old code
    #return min_slope, angles[index_min]
    return max_slope, angles[index_max]


def find_node_in_direction_of_max_slope_d4(self, u, node_id):
    """
    This method is exactly the same as find_node_in_direction_of_max_slope
    except that this method only considers nodes that are connected by links,
    or in otherwords, in the 0, 90, 180 and 270 directions.
    
        This method calculates the slopes (-dz/dx) in u across all 4 faces of 
        the cell with ID node_id. 
        It then returns the node ID in the direction of the steepest 
        (most positive) of these values,  i.e., this is a 
        D8 algorithm. Slopes downward from the cell are reported as positive.
        Based on code from DH, modified by NG, 6/2013
        
        This doesn't deal with the fixed gradient boundary condition.  
        NG is still confused about that one.
        
        NMG Update.  This is super clumsy. 
        
        DEJH update: Gets confused for the lowest node if w/i grid
        (i.e., closed)- will return a higher neighbour, when it should
        return a null index ->  Now returns -1.
    """
    #We have poor functionality if these are closed boundary nodes! 
    neighbor_nodes = self.get_neighbor_list(node_id)
    neighbor_nodes.sort()
    #print 'Node is internal: ', self.is_interior(cell_id)
    #print 'Neighbor cells: ', neighbor_cells
    slopes = []
    for a in neighbor_nodes:
        if self.node_status[a] != INACTIVE_BOUNDARY:
            single_slope = (u[node_id] - u[a])/self.dx
        else:
            single_slope = -9999
        #print 'cell id: ', cell_id
        #print 'neighbor id: ', a
        #print 'status: ', self.node_status[a]
        #print 'cell, neighbor are internal: ', self.is_interior(cell_id), self.is_interior(a)
        #print 'cell elev: ', u[cell_id]
        #print 'neighbor elev: ', u[a]
        #print single_slope
        if not numpy.isnan(single_slope): #This should no longer be necessary, but retained in case
            slopes.append(single_slope)
        else:
            print 'NaNs present in the grid!'

    #print 'Slopes list: ', slopes
    if slopes:
        max_slope, index_max = max((max_slope, index_max) for (index_max, max_slope) in enumerate(slopes))
    else:
        print u
        print 'Returning NaN angle and direction...'
        max_slope = numpy.nan
        index_max = 4
    
    #all_neighbor_nodes=numpy.concatenate((neighbor_nodes,diagonal_nodes))
    #print 'all_neighbor_cells ', all_neighbor_cells
    
    #Final check to  allow correct handling of internally draining nodes; DEJH Aug 2013.
    #This remains extremely ad-hoc. An internal node points to itself, but this should never
    #be used to actually route flow. In flow_accumulation, there is an explicit check that flow
    #is not routed to yourself.
    steepest_node = neighbor_nodes[index_max]
    #...now if a node is the lowest thing, this method returns -1, not a neighbor:
    if u[steepest_node] > u[node_id]:
        steepest_node=-1
    
    return steepest_node


def calculate_max_gradient_across_node_d4(self, u, cell_id):
    """
    This method calculates the gradients in u across all 4 faces of the 
    cell with ID cell_id. It then returns 
    the steepest (most negative) of these values, followed by its dip 
    direction (e.g.: 90 180). i.e., this is a D4 algorithm. Slopes 
    downward from the cell are reported as positive.
    
    Note that this is exactly the same as calculate_max_gradient_across_node
    except that this is d4, and the other is d8.
    
    This code is actually calculating slopes, not gradients.  
    The max gradient is the most negative, but the max slope is the most
    positive.  So, this was updated to return the max value, not the 
    min.
    """
    node_id = self.node_index_at_cells[cell_id]
    neighbor_nodes = self.get_neighbor_list(node_id)

    grads = (u[node_id] - u[neighbor_nodes]) / self.node_spacing
    ind = np.argmax(grads)
    _ANGLES = (90., 0., 270., 180.)
    return grads[ind], _ANGLES[ind]

    #We have poor functionality if these are edge cells! Needs an exception
    neighbor_cells = self.get_neighbor_list(cell_id)
    neighbor_cells.sort()
    #print 'Node is internal: ', self.is_interior(cell_id)
    #print 'Neighbor cells: ', neighbor_cells

    slopes = []
    for a in neighbor_cells:
        #ng I think this is actually slope as defined by a geomorphologist,
        #that is -dz/dx and not the gradient (dz/dx)
        if self.node_status[a] != INACTIVE_BOUNDARY:
            single_slope = (u[cell_id] - u[a])/self._dx
        else:
            single_slope = -9999
        #single_slope = (u[cell_id] - u[a])/self._dx
        #print 'cell id: ', cell_id
        #print 'neighbor id: ', a
        #print 'cell, neighbor are internal: ', self.is_interior(cell_id), self.is_interior(a)
        #print 'cell elev: ', u[cell_id]
        #print 'neighbor elev: ', u[a]
        #print single_slope
        #if not np.isnan(single_slope): #This should no longer be necessary, but retained in case
        #    slopes.append(single_slope)
        #else:
        #    print 'NaNs present in the grid!'
        slopes.append(single_slope)
            
    #print 'Slopes list: ', slopes
    #ng thinks that the maximum slope should be found here, not the 
    #minimum slope, old code commented out.  New code below it.
    #if slopes:
    #    min_slope, index_min = min((min_slope, index_min) for (index_min, min_slope) in enumerate(slopes))
    #else:
    #    print u
    #    print 'Returning NaN angle and direction...'
    #    min_slope = np.nan
    #    index_min = 8
    if slopes:
        max_slope, index_max = max((max_slope, index_max) for (index_max, max_slope) in enumerate(slopes))
    else:
        print u
        print 'Returning NaN angle and direction...'
        max_slope = np.nan
        index_max = 4
        
    angles = [180., 270., 90., 0., np.nan] #This is inefficient
    
    #ng commented out old code
    #return min_slope, angles[index_min]
    return max_slope, angles[index_max]
