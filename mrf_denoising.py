# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:35:26 2017

@author: carmonda
"""
import sys
from scipy import misc, imageio
import matplotlib.pyplot as plt
import numpy as np

PLOT = True
noise_segment = (15, 18, 3, 17)


class Vertex(object):
    def __init__(self, name='', y=None, neighs=None, in_msgs=None, observed=True):
        self._name = name
        self._y = y  # original pixel
        if neighs == None:
            neighs = set()  # set of neighbour nodes
        if in_msgs == None:
            in_msgs = {}  # dictionary mapping neighbours to their messages
        self._neighs = neighs
        self._in_msgs = in_msgs

    def add_neigh(self, vertex):
        self._neighs.add(vertex)

    def rem_neigh(self, vertex):
        self._neighs.remove(vertex)

    def get_belief(self):
        # TODO
        return

    def snd_msg(self, neigh):
        """ Combines messages from all other neighbours
            to propagate a message to the neighbouring Vertex 'neigh'.
        """
        # TODO
        return

    def __str__(self):
        ret = "Name: " + self._name
        ret += "\nNeighbours:"
        neigh_list = ""
        for n in self._neighs:
            neigh_list += " " + n._name
        ret += neigh_list
        return ret


class Graph(object):
    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary is given, an empty dict will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self._graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph"""
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
        if (type(v1) == Vertex and type(v2) == Vertex):
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


def is_observed(row, col):  # helper function for deciding which pixels are in the noisy segment.
    """
    Returns True/False by whether pixel at (row,col) was in the noisy segment or not.
    """
    x1, x2, y1, y2 = noise_segment

    def in_rect(row, col, x1, x2, y1, y2):
        if (row < x1 or row > x2):
            return False
        if (col < y1 or col > y2):
            return False
        return True

    return not (in_rect(row, col, x1, x2, y1, y2))


def build_grid_graph(n, m, img_mat):
    """ Builds an nxm grid graph, with vertex values corresponding to pixel intensities.
    n: num of rows
    m: num of columns
    img_mat = np.ndarray of shape (n,m) of pixel intensities
    
    returns the Graph object corresponding to the grid
    """
    V = []
    g = Graph()
    # add vertices:
    for i in range(n * m):
        row, col = (i // m, i % m)
        v = Vertex(name="v" + str(i), y=img_mat[row][col], observed=is_observed(row, col))
        g.add_vertex(v)
        if ((i % m) != 0):  # has left edge
            g.add_edge((v, V[i - 1]))
        if (i >= m):  # has up edge
            g.add_edge((v, V[i - m]))
        V += [v]
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
        mat[row][col] = 2017  # you should change this of course
    return mat


def main():
    # begin:
    if len(sys.argv) < 3:
        print 'Please specify input and output file names.'
        exit(0)
    # load image:
    in_file_name = sys.argv[1]
    image = imageio.imread(in_file_name + '.png')
    n, m = image.shape

    # binarize the image.
    image = image.astype(np.float32)
    image[image < 128] = -1.
    image[image > 127] = 1.
    if PLOT:
        plt.imshow(image)
        plt.show()

    # build grid:
    g = build_grid_graph(n, m, image)

    # process grid:

    # here you should do stuff to recover the image...
    # TODO
    # convert grid to image: 
    infered_img = grid2mat(g, n, m)
    if PLOT:
        plt.imshow(infered_img)
        plt.show()

    # save result to output file
    out_file_name = sys.argv[2]
    misc.toimage(infered_img).save(out_file_name + '.png')


if __name__ == "__main__":
    main()
