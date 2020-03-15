# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:35:26 2017
@author: carmonda
"""
#%%
import sys
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

PLOT = True
ALPHA = 0.5
BETA = 0.5

#%%
class Vertex(object):
	def __init__(self, name='', y=None, neighs=None, in_msgs=None):
		self._name = name
		self._y = y # original pixel
		if(neighs == None): neighs = set() # set of neighbour nodes
		if(in_msgs==None): in_msgs = {} # dictionary mapping neighbours to their messages
		# key - vertex.name , value = list of size 2 contains [log(m(ij)(-1),log(m(ij)(1)]
		self._neighs = neighs
		self._in_msgs = in_msgs
	def add_neigh(self,vertex):
		self._neighs.add(vertex)
		self._in_msgs[vertex._name]=[0,0]

	def rem_neigh(self,vertex):
		self._neighs.remove(vertex)
	def get_belief(self):
		# value of x_i
		result = np.sum(self._in_msgs.values(),axis=0).astype('float64')
		result+=np.array([-1.,1.])*ALPHA*self._y
		return 2*np.argmax(result)-1
		# TODO implement this
	def snd_msg(self,neigh):
		my_msgs= [val for key,val in self._in_msgs.items() if key!=neigh._name]
		# print my_msgs,self._name
		# exit(0)
		result = np.sum(my_msgs,axis=0).astype('float64')
		# xj=[-1.,1.]
		result+=np.array([-1.,1.])*ALPHA*self._y
		# xi=[-1.,1.].T
		result=result+np.array([-1.,1.]).reshape((-1,1))*BETA
		# print np.max(result,axis=0)
		result=np.max(result,axis=0)
		neigh._in_msgs[self._name]=np.exp(result)/np.sum(np.exp(result))
		# neigh._in_msgs[self._name]=result/np.sum((result))
		# return 2*np.argmax(result) -1
		# in_msgs.key(neigh)
		""" Combines messages from all other neighbours
			to propagate a message to the neighbouring Vertex 'neigh'.
		"""


	def __str__(self):
		ret = "Name: "+self._name
		ret += "\nNeighbours:"
		neigh_list = ""
		for n in self._neighs:
			neigh_list += " "+n._name
		ret+= neigh_list
		return ret

	__ne__ = lambda self, other: int(self._name[1:]) != int(other._name[1:])
	__lt__ = lambda self, other: int(self._name[1:]) < int(other._name[1:])
	__le__ = lambda self, other: int(self._name[1:]) <= int(other._name[1:])
	__gt__ = lambda self, other: int(self._name[1:]) > int(other._name[1:])
	__ge__ = lambda self, other: int(self._name[1:]) >= int(other._name[1:])

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
		return self._generate_edges()
	def add_vertex(self, vertex):
		""" If the vertex "vertex" is not in
			self._graph_dict, a key "vertex" with an empty
			list as a value is added to the dictionary.
			Otherwise nothing has to be done.
		"""
		if vertex not in self._graph_dict:
			self._graph_dict[vertex]=[]
	def add_edge(self,edge):
		""" assumes that edge is of type set, tuple, or list;
			between two vertices can be multiple edges.
		"""
		edge = set(edge)
		(v1,v2) = tuple(edge)
		if v1 in self._graph_dict:
			self._graph_dict[v1].append(v2)
		else:
			self._graph_dict[v1] = [v2]
		# if using Vertex class, update data:
		if(type(v1)==Vertex and type(v2)==Vertex):
			v1.add_neigh(v2)
			v2.add_neigh(v1)
	def _generate_edges(self):
		""" A static method generating the edges of the
			graph "graph". Edges are represented as sets
			with one or two vertices
		"""
		e = []
		for v in self._graph_dict:
			for neigh in self._graph_dict[v]:
				if {neigh,v} not in e:
					e.append({v,neigh})
		return e
	def __str__(self):
		res = "V: "
		res+=" ".join([str(k) for k in self._graph_dict])
		 # + " "
		res+= "\nE: "
		for edge in self._generate_edges():
			res+= str(edge) + " "
		return res

	def infer(self):
		for v in sorted(self.vertices()):
			# print v
			for u in v._neighs:
				v.snd_msg(u)


def build_grid_graph(n,m,img_mat):
	""" Builds an nxm grid graph, with vertex values corresponding to pixel intensities.
	n: num of rows
	m: num of columns
	img_mat = np.ndarray of shape (n,m) of pixel intensities
	returns the Graph object corresponding to the grid
	"""
	V = []
	g = Graph()
	# add vertices:
	for i in range(n*m):
		row,col = (i//m,i%m)
		v = Vertex(name="v"+str(i), y=img_mat[row][col])
		g.add_vertex(v)
		if((i%m)!=0): # has left edge
			g.add_edge((v,V[i-1]))
		if(i>=m): # has up edge
			g.add_edge((v,V[i-m]))
		V += [v]
	return g

# TODO calculate x_star
def grid2mat(grid,n,m):
	""" convertes grid graph to a np.ndarray
	n: num of rows
	m: num of columns
	returns: np.ndarray of shape (n,m)
	"""
	mat = np.zeros((n,m))
	l = grid.vertices() # list of vertices
	for v in l:
		i = int(v._name[1:])

		row,col = (i//m,i%m)
		mat[row][col] = v.get_belief() # TODO   you should change this of course
	return mat

#
def main():
	# begin:

	# if len(sys.argv) < 3: #TODO # FIXME:
	# if True:
	if len(sys.argv) < 3: #TODO # FIXME:
	# if True:
		in_file_name = "digit8_bn"
		out_file_name = "output"
		# print 'Please specify input and output file names.'
		# exit(0)
	else:
		in_file_name = sys.argv[1]
		out_file_name = sys.argv[2]
	# load image:
	image = misc.imread(in_file_name + '.png')
	n, m = image.shape

	# binarize the image.
	image = image.astype(np.float32)
	image[image<128] = -1.
	image[image>127] = 1.
	if PLOT:
		plt.imshow(image)
		plt.show()

	# build grid:
	g = build_grid_graph(n, m, image)
	prev=grid2mat(g, n, m)
	curr=np.zeros_like(prev)
	while not np.sum(np.equal(curr,prev)):
		# print "hi"
		g.infer()
		prev=curr
		curr=grid2mat(g, n, m)

	# process grid:
	# here you should do stuff to recover the image...

	# convert grid to image:
	infered_img = grid2mat(g, n, m)
	if PLOT:
		plt.imshow(infered_img)
		plt.show()

	# save result to output file

	misc.toimage(infered_img).save(out_file_name + '.png')


if __name__ == "__main__":
	main()
	# pass