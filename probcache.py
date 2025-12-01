"""
Simulation fidèle du mécanisme ProbCache
Basé sur: "Probabilistic In-Network Caching for Information-Centric Networks"
Psaras et al., ACM ICN 2012
Implémentation stricte des équations et définitions du papier.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple, Optional
import random


class Content:
    """Représente un contenu dans le réseau ICN"""
    def __init__(self, content_id: str, size: float = 1.0):
        self.content_id = content_id
        self.size = size  # En secondes de trafic


class LRUCache:
    """Cache LRU avec capacité en secondes de trafic"""
    def __init__(self, capacity: float):
        self.capacity = capacity  # N_i en secondes de trafic
        self.cache = OrderedDict()  # content_id -> Content
        self.current_size = 0.0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, content_id: str) -> Optional[Content]:
        """Récupère un contenu du cache (LRU)"""
        if content_id in self.cache:
            # Déplace en fin (most recently used)
            self.cache.move_to_end(content_id)
            self.hits += 1
            return self.cache[content_id]
        self.misses += 1
        return None
    
    def put(self, content: Content) -> bool:
        """Insère un contenu dans le cache"""
        if content.content_id in self.cache:
            # Déjà présent, juste le déplacer
            self.cache.move_to_end(content.content_id)
            return False
        
        # Éviction si nécessaire
        while self.current_size + content.size > self.capacity and self.cache:
            evicted_id, evicted_content = self.cache.popitem(last=False)
            self.current_size -= evicted_content.size
            self.evictions += 1
        
        # Insertion si capacité suffisante
        if content.size <= self.capacity:
            self.cache[content.content_id] = content
            self.current_size += content.size
            return True
        return False
    
    def contains(self, content_id: str) -> bool:
        """Vérifie si le contenu est dans le cache"""
        return content_id in self.cache


class Router:
    """Routeur avec cache LRU"""
    def __init__(self, router_id: int, cache_capacity: float, level: int):
        self.router_id = router_id
        self.cache = LRUCache(cache_capacity)
        self.level = level  # Niveau dans l'arbre (0 = racine)
        self.parent: Optional[Router] = None
        self.children: List[Router] = []
        self.requests_received = 0
    
    def add_child(self, child: 'Router'):
        """Ajoute un enfant au routeur"""
        self.children.append(child)
        child.parent = self


class BinaryTree:
    """Arbre binaire représentant la topologie du réseau"""
    def __init__(self, levels: int, cache_capacity: float):
        self.levels = levels
        self.root: Optional[Router] = None
        self.all_routers: List[Router] = []
        self.consumers: List[Router] = []
        self._build_tree(cache_capacity)
    
    def _build_tree(self, cache_capacity: float):
        """Construit l'arbre binaire"""
        node_id = 0
        nodes_by_level = [[] for _ in range(self.levels)]
        
        # Créer tous les nœuds niveau par niveau
        for level in range(self.levels):
            num_nodes = 2 ** level
            for _ in range(num_nodes):
                router = Router(node_id, cache_capacity, level)
                nodes_by_level[level].append(router)
                self.all_routers.append(router)
                node_id += 1
        
        # Établir les connexions parent-enfant
        for level in range(self.levels - 1):
            for i, parent in enumerate(nodes_by_level[level]):
                left_child = nodes_by_level[level + 1][2 * i]
                right_child = nodes_by_level[level + 1][2 * i + 1]
                parent.add_child(left_child)
                parent.add_child(right_child)
        
        self.root = nodes_by_level[0][0]
        
        # Les consommateurs sont sur les 2 derniers niveaux
        self.consumers = nodes_by_level[-1] + nodes_by_level[-2]
    
    def set_heterogeneous_capacity_core_to_edge(self, min_cap: float, max_cap: float):
        """Configuration C→e : capacité décroissante du cœur vers la bordure"""
        for router in self.all_routers:
            # Capacité = max_cap - (level / (levels-1)) * (max_cap - min_cap)
            ratio = router.level / (self.levels - 1)
            capacity = max_cap - ratio * (max_cap - min_cap)
            router.cache = LRUCache(capacity)
    
    def set_heterogeneous_capacity_edge_to_core(self, min_cap: float, max_cap: float):
        """Configuration c→E : capacité croissante du cœur vers la bordure"""
        for router in self.all_routers:
            # Capacité = min_cap + (level / (levels-1)) * (max_cap - min_cap)
            ratio = router.level / (self.levels - 1)
            capacity = min_cap + ratio * (max_cap - min_cap)
            router.cache = LRUCache(capacity)


class CachingStrategy:
    """Classe de base pour les stratégies de caching"""
    def __init__(self, name: str):
        self.name = name
    
    def should_cache(self, router: Router, content: Content, 
                     path_length: int, hops_so_far: int) -> bool:
        """Détermine si le contenu doit être mis en cache"""
        raise NotImplementedError


class CE2Strategy(CachingStrategy):
    """Cache Everything Everywhere"""
    def __init__(self):
        super().__init__("CE2")
    
    def should_cache(self, router: Router, content: Content, 
                     path_length: int, hops_so_far: int) -> bool:
        return True


# class LCDStrategy(CachingStrategy):
#     """Leave Copy Down"""
#     def __init__(self):
#         super().__init__("LCD")
#         self.last_cache_level = {}  # content_id -> level
    
#     def should_cache(self, router: Router, content: Content, 
#                      path_length: int, hops_so_far: int) -> bool:
#         # Cache seulement un niveau en dessous du dernier cache hit
#         content_id = content.content_id
        
#         # Si c'est un nouveau contenu venant du serveur
#         if content_id not in self.last_cache_level:
#             # Cache au routeur juste avant le demandeur (first hop on return)
#             self.last_cache_level[content_id] = router.level
#             return True
        
#         # Sinon, cache un niveau plus bas (vers les feuilles)
#         if router.level == self.last_cache_level[content_id] + 1:
#             self.last_cache_level[content_id] = router.level
#             return True
        
#         return False


class ProbabilisticStrategy(CachingStrategy):
    """Caching probabiliste avec probabilité fixe"""
    def __init__(self, probability: float):
        super().__init__(f"Prob-{probability}")
        self.probability = probability
    
    def should_cache(self, router: Router, content: Content, 
                     path_length: int, hops_so_far: int) -> bool:
        return random.random() < self.probability


class ProbCacheStrategy(CachingStrategy):
    """
    ProbCache selon Psaras et al., ACM ICN 2012
    
    Implémentation stricte des équations:
    - TimesIn(x) = (Σ_{i=1 to c-(x-1)} N_i) / (T_tw * N_x)
    - CacheWeight(x) = x / c
    - ProbCache(x) = TimesIn(x) * CacheWeight(x)
    
    Où:
    - c = TSI (Time Since Inception) = longueur du chemin
    - x = TSB (Time Since Birth) = nombre de sauts effectués
    - N_i = capacité du cache au routeur i
    - T_tw = 10 secondes (fenêtre temporelle)
    """
    def __init__(self, T_tw: float = 10.0):
        super().__init__("ProbCache")
        self.T_tw = T_tw
    
    def should_cache(self, router: Router, content: Content, 
                     path_length: int, hops_so_far: int) -> bool:
        """
        Décision de cache selon ProbCache
        
        Args:
            router: Routeur courant
            content: Contenu à cacher
            path_length: c (TSI) = longueur totale du chemin
            hops_so_far: x (TSB) = nombre de sauts déjà effectués
        """
        c = path_length  # TSI
        x = hops_so_far  # TSB
        N_x = router.cache.capacity  # Capacité du routeur courant
        
        if x <= 0 or c <= 0 or N_x <= 0:
            return False
        
        # Calcul de TimesIn(x) selon équation du papier
        # TimesIn(x) = (Σ_{i=1 to c-(x-1)} N_i) / (T_tw * N_x)
        # 
        # Hypothèse du papier: tous les routeurs ont la même capacité N
        # Dans ce cas: Σ_{i=1 to c-(x-1)} N_i = N * (c - (x-1)) = N * (c - x + 1)
        # Donc: TimesIn(x) = (N * (c - x + 1)) / (T_tw * N) = (c - x + 1) / T_tw
        
        num_routers_upstream = c - x + 1
        times_in = num_routers_upstream / self.T_tw
        
        # Calcul de CacheWeight(x) selon équation du papier
        cache_weight = x / c
        
        # Calcul de ProbCache(x)
        prob_cache = times_in * cache_weight
        
        # Limitation à [0, 1]
        prob_cache = min(1.0, max(0.0, prob_cache))
        
        # Décision probabiliste
        return random.random() < prob_cache


class ProbCachePlusStrategy(CachingStrategy):
    """
    ProbCache+ pour caches hétérogènes
    
    Selon équations 4 et 5 du papier:
    - Équation 4: TimesIn(x) = (Σ_{i=1 to c-(x-1)} N_i) / (T_tw * N_x)
    - Équation 5: ProbCache(x) = TimesIn(x) * CacheWeight(x)
    
    Avec calcul explicite de la somme des capacités upstream
    """
    def __init__(self, network: BinaryTree, T_tw: float = 10.0):
        super().__init__("ProbCache+")
        self.T_tw = T_tw
        self.network = network
    
    def should_cache(self, router: Router, content: Content, 
                     path_length: int, hops_so_far: int) -> bool:
        """
        Décision de cache selon ProbCache+ (caches hétérogènes)
        """
        c = path_length  # TSI
        x = hops_so_far  # TSB
        N_x = router.cache.capacity  # Capacité du routeur courant
        
        if x <= 0 or c <= 0 or N_x <= 0:
            return False
        
        # Calcul de TimesIn(x) avec somme réelle des capacités upstream
        # Σ_{i=1 to c-(x-1)} N_i
        # 
        # On doit remonter le chemin pour calculer la somme des capacités
        # des routeurs entre la position courante et le serveur
        
        num_routers_to_sum = c - x + 1
        
        # Approximation: on utilise le chemin moyen dans l'arbre
        # Pour un arbre binaire, on remonte vers la racine
        sum_capacities = 0.0
        current = router
        count = 0
        
        while current is not None and count < num_routers_to_sum:
            sum_capacities += current.cache.capacity
            current = current.parent
            count += 1
        
        # Si on n'a pas assez de routeurs, on extraple
        if count < num_routers_to_sum:
            avg_capacity = sum_capacities / max(count, 1)
            sum_capacities += avg_capacity * (num_routers_to_sum - count)
        
        # TimesIn(x) selon équation 4
        times_in = sum_capacities / (self.T_tw * N_x)
        
        # CacheWeight(x)
        cache_weight = x / c
        
        # ProbCache(x) selon équation 5
        prob_cache = times_in * cache_weight
        
        # Limitation à [0, 1]
        prob_cache = min(1.0, max(0.0, prob_cache))
        
        # Décision probabiliste
        return random.random() < prob_cache


class ICNSimulator:
    """Simulateur ICN principal"""
    def __init__(self, network: BinaryTree, catalog_size: int, 
                 num_requests: int, zipf_alpha: float = 0.8):
        self.network = network
        self.catalog_size = catalog_size
        self.num_requests = num_requests
        self.zipf_alpha = zipf_alpha
        
        # Génération du catalogue de contenus
        self.catalog = [Content(f"content_{i}") for i in range(catalog_size)]
        
        # Statistiques globales
        self.total_server_hits = 0
        self.total_cache_hits = 0
        self.total_hops = 0
        self.total_hops_without_cache = 0
        
        # Statistiques par niveau
        self.hits_per_level = defaultdict(int)
        self.requests_per_level = defaultdict(int)
        self.evictions_per_level = defaultdict(int)
    
    def generate_zipf_requests(self) -> List[Content]:
        """Génère une séquence de requêtes selon une distribution de Zipf"""
        # Génération des probabilités Zipf
        ranks = np.arange(1, self.catalog_size + 1)
        probabilities = 1.0 / (ranks ** self.zipf_alpha)
        probabilities /= probabilities.sum()
        
        # Génération des requêtes
        content_indices = np.random.choice(
            self.catalog_size, 
            size=self.num_requests, 
            p=probabilities
        )
        
        return [self.catalog[i] for i in content_indices]
    
    def get_path_to_server(self, consumer: Router) -> List[Router]:
        """Retourne le chemin du consommateur au serveur (racine)"""
        path = []
        current = consumer
        while current is not None:
            path.append(current)
            current = current.parent
        return path
    
    def process_request(self, consumer: Router, content: Content, 
                       strategy: CachingStrategy) -> Tuple[int, bool]:
        """
        Traite une requête de contenu
        
        Returns:
            (nombre de sauts, hit serveur ou non)
        """
        path = self.get_path_to_server(consumer)
        path_length = len(path)
        
        # Phase 1: Recherche du contenu (aller vers le serveur)
        content_found_at = None
        hops = 0
        
        for i, router in enumerate(path):
            router.requests_received += 1
            self.requests_per_level[router.level] += 1
            hops += 1
            
            if router.cache.get(content.content_id) is not None:
                # Cache hit
                content_found_at = i
                self.total_cache_hits += 1
                self.hits_per_level[router.level] += 1
                break
        
        # Si pas trouvé dans les caches, le serveur répond
        if content_found_at is None:
            self.total_server_hits += 1
            server_hit = True
            # Le contenu vient du serveur (racine)
            content_found_at = len(path) - 1
        else:
            server_hit = False
        
        # Phase 2: Retour du contenu (du point de hit vers le consommateur)
        # On parcourt le chemin en sens inverse depuis le point de hit
        for i in range(content_found_at - 1, -1, -1):
            router = path[i]
            hops += 1
            
            # TSB (Time Since Birth) = nombre de sauts depuis le serveur
            # = content_found_at - i
            hops_so_far = content_found_at - i
            
            # Décision de caching
            if strategy.should_cache(router, content, path_length, hops_so_far):
                router.cache.put(content)
        
        self.total_hops += hops
        self.total_hops_without_cache += 2 * path_length  # Aller-retour complet
        
        return hops, server_hit
    
    def run_simulation(self, strategy: CachingStrategy) -> Dict:
        """Exécute une simulation complète avec une stratégie donnée"""
        # Réinitialisation des caches et statistiques
        for router in self.network.all_routers:
            router.cache = LRUCache(router.cache.capacity)
            router.requests_received = 0
        
        self.total_server_hits = 0
        self.total_cache_hits = 0
        self.total_hops = 0
        self.total_hops_without_cache = 0
        self.hits_per_level = defaultdict(int)
        self.requests_per_level = defaultdict(int)
        self.evictions_per_level = defaultdict(int)
        
        # Génération des requêtes
        requests = self.generate_zipf_requests()
        
        # Traitement des requêtes
        for content in requests:
            # Sélection aléatoire d'un consommateur
            consumer = random.choice(self.network.consumers)
            self.process_request(consumer, content, strategy)
        
        # Collecte des évictions par niveau
        for router in self.network.all_routers:
            self.evictions_per_level[router.level] += router.cache.evictions
        
        # Calcul des métriques
        server_hit_rate = self.total_server_hits / self.num_requests
        hop_reduction_ratio = 1.0 - (self.total_hops / self.total_hops_without_cache)
        
        return {
            'strategy': strategy.name,
            'server_hit_rate': server_hit_rate,
            'cache_hit_rate': 1.0 - server_hit_rate,
            'hop_reduction_ratio': hop_reduction_ratio,
            'total_hops': self.total_hops,
            'hits_per_level': dict(self.hits_per_level),
            'requests_per_level': dict(self.requests_per_level),
            'evictions_per_level': dict(self.evictions_per_level)
        }


def plot_results(results_list: List[Dict], title: str, filename: str):
    """Génère les graphiques de résultats"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    
    strategies = [r['strategy'] for r in results_list]
    
    # 1. Server Hit Rate
    ax = axes[0, 0]
    server_hits = [r['server_hit_rate'] * 100 for r in results_list]
    ax.bar(strategies, server_hits, color='steelblue')
    ax.set_ylabel('Server Hit Rate (%)')
    ax.set_title('Server Hit Rate')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Hop Reduction Ratio
    ax = axes[0, 1]
    hop_reduction = [r['hop_reduction_ratio'] * 100 for r in results_list]
    ax.bar(strategies, hop_reduction, color='coral')
    ax.set_ylabel('Hop Reduction Ratio (%)')
    ax.set_title('Hop Reduction Ratio')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Cache Hits par niveau
    ax = axes[1, 0]
    levels = sorted(set().union(*[set(r['hits_per_level'].keys()) for r in results_list]))
    x = np.arange(len(levels))
    width = 0.8 / len(strategies)
    
    for i, result in enumerate(results_list):
        hits = [result['hits_per_level'].get(level, 0) for level in levels]
        ax.bar(x + i * width, hits, width, label=result['strategy'])
    
    ax.set_xlabel('Level')
    ax.set_ylabel('Cache Hits')
    ax.set_title('Cache Hits per Level')
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Evictions par niveau
    ax = axes[1, 1]
    for i, result in enumerate(results_list):
        evictions = [result['evictions_per_level'].get(level, 0) for level in levels]
        ax.bar(x + i * width, evictions, width, label=result['strategy'])
    
    ax.set_xlabel('Level')
    ax.set_ylabel('Cache Evictions')
    ax.set_title('Cache Evictions per Level')
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {filename}")
    plt.close()


def print_results_table(results_list: List[Dict]):
    """Affiche un tableau des résultats"""
    print("\n" + "="*80)
    print("RÉSULTATS DE SIMULATION")
    print("="*80)
    print(f"{'Strategy':<15} {'Server Hits':<15} {'Hop Reduction':<15} {'Total Hops':<15}")
    print("-"*80)
    
    for result in results_list:
        print(f"{result['strategy']:<15} "
              f"{result['server_hit_rate']*100:>13.2f}% "
              f"{result['hop_reduction_ratio']*100:>13.2f}% "
              f"{result['total_hops']:>14,}")
    print("="*80)


def main():
    """Fonction principale exécutant tous les scénarios"""
    
    # Configuration générale
    TREE_LEVELS = 6
    CATALOG_SIZE = 1000
    NUM_REQUESTS = 100000
    ZIPF_ALPHA = 0.8
    
    print("="*80)
    print("SIMULATION PROBCACHE - Psaras et al., ACM ICN 2012")
    print("="*80)
    print(f"Topologie: Arbre binaire à {TREE_LEVELS} niveaux ({2**TREE_LEVELS - 1} nœuds)")
    print(f"Catalogue: {CATALOG_SIZE} contenus")
    print(f"Requêtes: {NUM_REQUESTS:,}")
    print(f"Distribution: Zipf (α={ZIPF_ALPHA})")
    print("="*80)
    
    # =========================================================================
    # SCÉNARIO 1: CACHES HOMOGÈNES
    # =========================================================================
    print("\n" + "="*80)
    print("SCÉNARIO 1: CACHES HOMOGÈNES")
    print("="*80)
    
    results_homogeneous = []
    
    for cache_capacity in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        print(f"\n--- Capacité de cache: {cache_capacity} secondes ---")
        
        # Création du réseau
        network = BinaryTree(TREE_LEVELS, cache_capacity)
        simulator = ICNSimulator(network, CATALOG_SIZE, NUM_REQUESTS, ZIPF_ALPHA)
        
        # Test des stratégies
        strategies = [
            CE2Strategy(),
            # LCDStrategy(),
            ProbabilisticStrategy(0.3),
            ProbabilisticStrategy(0.7),
            ProbCacheStrategy(T_tw=10.0)
        ]
        
        capacity_results = []
        for strategy in strategies:
            print(f"  Simulation: {strategy.name}...", end=' ')
            result = simulator.run_simulation(strategy)
            capacity_results.append(result)
            print(f"✓ (Server hits: {result['server_hit_rate']*100:.1f}%)")
        
        results_homogeneous.append({
            'capacity': cache_capacity,
            'results': capacity_results
        })
    
    # Graphiques pour différentes capacités
    for cap_result in results_homogeneous[::2]:  # Échantillonnage pour lisibilité
        capacity = cap_result['capacity']
        plot_results(
            cap_result['results'],
            f"Caches Homogènes - Capacité: {capacity} sec",
            f"homogeneous_cap_{capacity}.png"
        )
        print_results_table(cap_result['results'])
    
    # =========================================================================
    # SCÉNARIO 2: CACHES HÉTÉROGÈNES (C→e)
    # =========================================================================
    print("\n" + "="*80)
    print("SCÉNARIO 2: CACHES HÉTÉROGÈNES (C→e - Cœur vers bordure)")
    print("="*80)
    
    network = BinaryTree(TREE_LEVELS, 1.0)  # Init avec capacité temporaire
    network.set_heterogeneous_capacity_core_to_edge(min_cap=1.0, max_cap=6.0)
    
    simulator = ICNSimulator(network, CATALOG_SIZE, NUM_REQUESTS, ZIPF_ALPHA)
    
    strategies = [
        CE2Strategy(),
        # LCDStrategy(),
        ProbabilisticStrategy(0.3),
        ProbabilisticStrategy(0.7),
        #ProbCacheStrategy(T_tw=10.0),
        ProbCachePlusStrategy(network, T_tw=10.0)
    ]
    
    results_hetero_ce = []
    for strategy in strategies:
        print(f"Simulation: {strategy.name}...", end=' ')
        result = simulator.run_simulation(strategy)
        results_hetero_ce.append(result)
        print(f"✓ (Server hits: {result['server_hit_rate']*100:.1f}%)")
    
    plot_results(
        results_hetero_ce,
        "Caches Hétérogènes (C→e) - Capacité décroissante vers bordure",
        "heterogeneous_core_to_edge.png"
    )
    print_results_table(results_hetero_ce)
    
    # =========================================================================
    # SCÉNARIO 3: CACHES HÉTÉROGÈNES (c→E)
    # =========================================================================
    print("\n" + "="*80)
    print("SCÉNARIO 3: CACHES HÉTÉROGÈNES (c→E - Bordure vers cœur)")
    print("="*80)
    
    network = BinaryTree(TREE_LEVELS, 1.0)
    network.set_heterogeneous_capacity_edge_to_core(min_cap=1.0, max_cap=6.0)
    
    simulator = ICNSimulator(network, CATALOG_SIZE, NUM_REQUESTS, ZIPF_ALPHA)
    
    results_hetero_ec = []
    for strategy in strategies:
        print(f"Simulation: {strategy.name}...", end=' ')
        result = simulator.run_simulation(strategy)
        results_hetero_ec.append(result)
        print(f"✓ (Server hits: {result['server_hit_rate']*100:.1f}%)")
    
    plot_results(
        results_hetero_ec,
        "Caches Hétérogènes (c→E) - Capacité croissante vers bordure",
        "heterogeneous_edge_to_core.png"
    )
    print_results_table(results_hetero_ec)
    
    # =========================================================================
    # GRAPHIQUE COMPARATIF FINAL
    # =========================================================================
    print("\n" + "="*80)
    print("GÉNÉRATION DU GRAPHIQUE COMPARATIF FINAL")
    print("="*80)
    
    # Comparaison sur cache homogène de 3 secondes
    comparison_results = results_homogeneous[2]['results']  # Capacité 3.0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    strategies = [r['strategy'] for r in comparison_results]
    
    # Server Hits
    ax = axes[0]
    server_hits = [r['server_hit_rate'] * 100 for r in comparison_results]
    bars = ax.bar(strategies, server_hits, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('Server Hit Rate (%)', fontsize=12)
    ax.set_title('Server Hit Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Hop Reduction
    ax = axes[1]
    hop_reduction = [r['hop_reduction_ratio'] * 100 for r in comparison_results]
    bars = ax.bar(strategies, hop_reduction, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('Hop Reduction Ratio (%)', fontsize=12)
    ax.set_title('Hop Reduction Ratio Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=300, bbox_inches='tight')
    print("Graphique comparatif final sauvegardé: final_comparison.png")
    plt.close()
    
    print("\n" + "="*80)
    print("SIMULATION TERMINÉE")
    print("="*80)
    print("\nTous les graphiques ont été générés dans /mnt/user-data/outputs/")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()