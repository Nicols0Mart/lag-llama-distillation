import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Define the area of interest
place = 'Brussels, Brussels-Capital, Belgium'

# Fetch the street network graph for the specified area
G = ox.graph_from_place(place, network_type='drive')

# Get positions of nodes for plotting
pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

# Number of time steps for the animation
T = 100

# Generate traffic flow parameters for each edge
edge_params = {}
for u, v, data in G.edges(data=True):
    base_flow = np.random.uniform(0.3, 0.7)
    amplitude = np.random.uniform(0.1, 0.3)
    frequency = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, 2 * np.pi)
    edge_params[(u, v)] = {'base_flow': base_flow, 'amplitude': amplitude,
                           'frequency': frequency, 'phase': phase}

# Generate traffic flow values over time for each edge
edge_flows = {}
t = np.arange(T)
for edge, params in edge_params.items():
    base_flow = params['base_flow']
    amplitude = params['amplitude']
    frequency = params['frequency']
    phase = params['phase']
    flows = base_flow + amplitude * np.sin(2 * np.pi * frequency * t / T + phase) + 0.05 * np.random.randn(T)
    flows = np.clip(flows, 0, 1)  # Ensure values are between 0 and 1
    edge_flows[edge] = flows

# Prepare the plot
fig, ax = plt.subplots(figsize=(20, 20))

# Prepare flows and edges for drawing
flows = [edge_flows[(u, v)][0] for u, v in G.edges()]  # Use the initial flow (t=0)
edges = list(G.edges())

# Normalize the edge flows for colormap
norm = Normalize(vmin=min(flows), vmax=max(flows))
cmap = plt.cm.jet
edge_colors = [cmap(norm(flow)) for flow in flows]

# Draw the network edges with the generated colors
nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=edge_colors,
                       edge_cmap=cmap, edge_vmin=0, edge_vmax=1, width=2)

# Create a ScalarMappable for the colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # ScalarMappable requires a set array to work

# Add the colorbar to the plot
fig.colorbar(sm, ax=ax, orientation='horizontal', label='Traffic Flow Intensity')

# Show the plot
plt.show()



# Function to animate the traffic flow
# def animate(i):
#     M = G.number_of_edges()
#     edge_colors = range(2, G.number_of_edges() + 2)
#     ax.clear()
#     # cax = div.append_axes('right', '5%', '5%')

#     flows = []
#     colors = []
#     edges = []
#     for u, v, data in G.edges(data=True):
#         flow = edge_flows[(u, v)][i]
#         flows.append(flow)
#         edges.append((u, v))
#         colors.append(flow)
#     edges = nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=colors,
#                            edge_cmap=cmap, edge_vmin=0, edge_vmax=1, width=2)
#     # edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
#     # for i in range(M):
#     #     edges[i].set_alpha(edge_alphas[i])
#     # nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10)
#     # pc = matplotlib.collections.PatchCollection(edges, cmap=cmap)
#     # pc.set_array(edge_colors)
#     # levels = np.linspace(0, 1, len(flow), endpoint = True)
#     # cf = ax.contourf(edges, vmax=1, vmin=0, levels=levels)
#     # cax.cla()
#     # fig.colorbar(cf, cax=cax)
#     ax.set_title(f'Time step {i}')
#     ax.set_axis_off()
#     # plt.colorbar(pc, ax=ax)

# # Create the animation
# ani = animation.FuncAnimation(fig, animate, frames=T, interval=100)

# # Display the animation (uncomment the following lines if running in Jupyter Notebook)
# # from IPython.display import HTML
# # HTML(ani.to_jshtml())

# # Save the animation as a GIF

# # plt.legend()


# ani.save('traffic_flow.gif', writer='pillow')

plt.show()