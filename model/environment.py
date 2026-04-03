"""
environment.py — Entorno de red social (v2, idéntico a v1).

Construye y actualiza un grafo NetworkX sobre el que
se propagan las interacciones entre agentes.

Compatible con Mesa 3.x: usa model.agents (AgentSet).
"""

import networkx as nx
import random


def build_small_world(agents, k: int = 4, p: float = 0.1) -> nx.Graph:
    """
    Red de mundo pequeño (Watts-Strogatz).

    Parámetros
    ----------
    agents : list
        Lista de agentes del modelo.
    k : int
        Número de vecinos iniciales.
    p : float
        Probabilidad de reconexión aleatoria.
    """
    agent_list = list(agents)
    n = len(agent_list)
    G = nx.watts_strogatz_graph(n, k, p)
    mapping = {i: agent_list[i].unique_id for i in range(n)}
    return nx.relabel_nodes(G, mapping)


def build_scale_free(agents, m: int = 2) -> nx.Graph:
    """Red libre de escala (Barabási-Albert)."""
    agent_list = list(agents)
    n = len(agent_list)
    G = nx.barabasi_albert_graph(n, m)
    mapping = {i: agent_list[i].unique_id for i in range(n)}
    return nx.relabel_nodes(G, mapping)


def build_random(agents, p: float = 0.05) -> nx.Graph:
    """Red aleatoria de Erdős-Rényi."""
    agent_list = list(agents)
    n = len(agent_list)
    G = nx.erdos_renyi_graph(n, p)
    mapping = {i: agent_list[i].unique_id for i in range(n)}
    return nx.relabel_nodes(G, mapping)


def get_neighbors(graph: nx.Graph, agent_id: int) -> list:
    """Retorna los IDs de los vecinos de un agente en el grafo."""
    if agent_id in graph:
        return list(graph.neighbors(agent_id))
    return []


def network_clustering(graph: nx.Graph) -> float:
    """Coeficiente de clustering promedio de la red."""
    return nx.average_clustering(graph)


def update_ties(graph: nx.Graph, agent, model, rewire_prob: float = 0.01):
    """
    Actualiza lazos del agente: elimina vínculos con agentes de muy
    baja reputación y añade nuevos con probabilidad baja.
    """
    agent_map = {a.unique_id: a for a in model.agents}
    neighbors = list(graph.neighbors(agent.unique_id))

    for nid in neighbors:
        neighbor = agent_map.get(nid)
        if neighbor and neighbor.reputation < 0.2:
            graph.remove_edge(agent.unique_id, nid)

    if random.random() < rewire_prob:
        candidates = [
            a.unique_id for a in model.agents
            if a.unique_id != agent.unique_id
            and not graph.has_edge(agent.unique_id, a.unique_id)
        ]
        if candidates:
            new_neighbor = random.choice(candidates)
            graph.add_edge(agent.unique_id, new_neighbor)
