# -*- coding: utf-8 -*-
import sys
from scipy import misc
import imageio
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

PLOT = True  # if true, main() will plot "before" and "after"

ALPHA = 1 # loyal to the observed pixel
BETA = 0.1  # smoothness term
SENSITIVITY_THRESHOLD = 0.01  # stop when the messages aren't changed more thant this threshold (l1 norm)

VALUES = [-1, 1]


def main():
    # begin:
    in_file_name, out_file_name = read_params()
    image = load_and_binarize_image_from_file(in_file_name)
    plot_image(image)
    # build grid:
    g = build_grid_graph(image)

    # process grid:
    g = process_grid(g)  # convert grid to image:
    infered_img = grid2mat(g, image.shape[0], image.shape[1])
    plot_image(infered_img)
    # save result to output file
    save_image(image=infered_img, out_file_name=out_file_name)


def read_params():
    if len(sys.argv) < 3:
        print ('Please specify input and output file names.')
        exit(0)
    return sys.argv[1], sys.argv[2]


def load_and_binarize_image_from_file(in_file_name):
    image = load_image(in_file_name)
    image = binarize_image(image)
    return image


def load_image(in_file_name):
    return imageio.imread(in_file_name + '.png')


def binarize_image(image):
    image = image.astype(np.float32)
    image[image < 128] = -1.
    image[image > 127] = 1.
    return image


def plot_image(image):
    if PLOT:
        plt.imshow(image)
        plt.show()


def build_grid_graph(img_mat):
    """ Builds an nxm grid graph, with vertex values corresponding to pixel intensities.
    n: num of rows
    m: num of columns
    img_mat = np.ndarray of shape (n,m) of pixel intensities

    returns the Graph object corresponding to the grid
    :rtype: Graph
    """
    n, m = img_mat.shape
    V = []
    g = Graph()
    # add vertices:
    for i in range(n * m):
        row, col = (i // m, i % m)
        y_value = img_mat[row][col]
        v = Vertex(name="v" + str(i), y=y_value)
        g.add_vertex(v)
        if ((i % m) != 0):  # has left edge
            g.add_edge((v, V[i - 1]))
        if (i >= m):  # has up edge
            g.add_edge((v, V[i - m]))
        V += [v]
    return g


def process_grid(g):
    max_delta_this_iteration = SENSITIVITY_THRESHOLD + 1
    while max_delta_this_iteration > SENSITIVITY_THRESHOLD:
        max_delta_this_iteration = 0.0
        for v in g.vertices():
            for neighbor in v._neighbors:
                old_message = neighbor.in_messages[v]
                new_message = v.send_message(neighbor)
                delta = message_distance(old_message, new_message)
                if delta > max_delta_this_iteration:
                    max_delta_this_iteration = delta
    return g

def grid2mat(grid, n, m):
    """ convertes grid graph to a np.ndarray
    n: num of rows
    m: num of columns

    returns: np.ndarray of shape (n,m)
    """
    mat = np.zeros((n, m))
    l = grid.vertices()  # list of vertices
    for v in l:
        i = int(v._name[1:])
        row, col = (i // m, i % m)
        mat[row][col] = v.get_belief()
    return mat


def save_image(image, out_file_name):
    misc.toimage(image).save(out_file_name + '.png')


class Graph(object):
    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary is given, an empty dict will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self._graph_dict = graph_dict

    def vertices(self):
        """
        returns the vertices of a graph

        :rtype: Iterable<Vertex>
        """
        return list(self._graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple, or list;
            between two vertices can be multiple edges.
        """
        edge = set(edge)
        (v1, v2) = tuple(edge)
        if v1 in self._graph_dict:
            self._graph_dict[v1].append(v2)
        else:
            self._graph_dict[v1] = [v2]
        # if using Vertex class, update data:
        if type(v1) == Vertex and type(v2) == Vertex:
            v1.add_neighbor(v2)
            v2.add_neighbor(v1)

    def generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one or two vertices
        """
        e = []
        for v in self._graph_dict:
            for neigh in self._graph_dict[v]:
                if {neigh, v} not in e:
                    e.append({v, neigh})
        return e

    def __str__(self):
        res = "V: "
        for k in self._graph_dict:
            res += str(k) + " "
        res += "\nE: "
        for edge in self.generate_edges():
            res += str(edge) + " "
        return res


class Vertex(object):
    def __init__(self, name='', y=None, neighbors=None, in_messages=None):
        self._name = name
        self._y = y  # original pixel
        if (neighbors == None):
            neighbors = set()  # set of neighbour nodes
        if (in_messages == None):
            in_messages = defaultdict(lambda: np.array([0,0])) # dictionary mapping neighbours to their messages
        self._neighbors = neighbors
        self.in_messages = in_messages

    def add_neighbor(self, vertex):
        self._neighbors.add(vertex)

    def remove_neighbor(self, vertex):
        self._neighbors.remove(vertex)

    def get_belief(self):
        m = np.array([xi * self._y * ALPHA for xi in VALUES])
        for neighbor in self._neighbors:
            m += self.in_messages[neighbor]
        index_of_maximal_xi = np.argmax(m)
        return 	VALUES[index_of_maximal_xi]

    def _compute_message(self, target_neighbor):
        # the message has 2 parts: one depending on xj and one not.
        independent_part = self._compute_independent_part_of_message(target_neighbor)

        # compute now the whole thing:
        options_to_send = self._compute_options_for_message(independent_part)
        message = np.max(options_to_send, 1)
        message -= np.math.log(np.sum(np.exp(message))) #NORMALIZE (in log space)
        return message

    def _compute_independent_part_of_message(self, target_neighbor):
        independent_part = np.array([xi * ALPHA * self._y for xi in VALUES])
        other_neighbors = set(self._neighbors)
        other_neighbors.remove(target_neighbor)
        for other_neighbor in other_neighbors:
            independent_part += self.in_messages[other_neighbor]
        return independent_part

    def _compute_options_for_message(self, independent_part):
        options_to_send = np.array([np.array([xi * BETA * xj for xi in VALUES]) + independent_part for xj in VALUES])
        return options_to_send

    def send_message(self, target_neighbor):
        """ Combines messages from all other neighbours
            to propagate a message to the neighbouring Vertex 'neigh'.
        """
        message = self._compute_message(target_neighbor=target_neighbor)
        target_neighbor.in_messages[self] = message
        return message

    def __str__(self):
        ret = "Name: " + self._name
        ret += "\nNeighbours:"
        neigh_list = ""
        for n in self._neighbors:
            neigh_list += " " + n._name
        ret += neigh_list
        return ret

    def __repr__(self):
        return self._name


def message_distance(m1, m2):  # l1 norm
    return sum(np.fabs(m1 - m2))


if __name__ == "__main__":
    main()

"""
phi(x_i, y_i) = exp(alpha * x_i * y_i)
phi(x_i, x_j) = exp(beta * x_i * x_j)

psi_ij(x_i, x_j) = phi(x_i, x_j)
psi_i(x_i) = phi(x_i, y_i)

m_ij (x_j) := max_xi [psi_i(x_i) * psi_ij(x_i, x_j) * \Prod_k\in N(i)\j [m_ki(xi)]]

log m_ij := max_xi [xi * (alpha * yi + beta * xj) + \Sum_k\in N(i)\j logm_ki(xi)]

at convergence:
Xi = argmax_xi [psi_i(xi) * \Prod_k\in N(i) [m_ki(xi)]]
log xi = argmax_xi [alpha * xi * yi + sum_k in N(i) logm_ki(xi)


normalize messages: for each m_ij:

m_ij(xi) = m_ij'(xi)/\sum_xi* m_ij(xi*)

so
logm_ij(xi) = logm_ij'(xi) - (log(sum_xi(e**logm_ij(xi))))

"""
