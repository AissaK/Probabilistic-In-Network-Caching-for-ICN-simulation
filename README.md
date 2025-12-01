# ProbCache Simulation - Information-Centric Networks

Faithful implementation of the **ProbCache** mechanism for ICN (Information-Centric Networks), based on the research paper:

> **"Probabilistic In-Network Caching for Information-Centric Networks"**  
> *Psaras, I., Chai, W. K., & Pavlou, G. (2012)*  
> ACM SIGCOMM Workshop on Information-Centric Networking (ICN '12)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Architecture](#-architecture)
- [Implemented Equations](#-implemented-equations)
- [Extending the Code](#-extending-the-code)
- [References](#-references)
- [License](#-license)

##  Overview

This simulation compares **5 caching strategies** for ICN networks:

1. **CE2** - Cache Everything Everywhere
2. **Prob-0.3** - Probabilistic caching (p=0.3)
3. **Prob-0.7** - Probabilistic caching (p=0.7)
4. **ProbCache** - Adaptive strategy (original paper)
5. **ProbCache+** - Variant for heterogeneous networks

> **Note**: The LCD (Leave Copy Down) strategy is commented out in the code due to potentially incorrect implementation producing unexpected results. Future work may revisit this implementation.

The simulator evaluates performance across **3 scenarios**:
- Homogeneous caches (identical capacities)
- Heterogeneous caches C→e (decreasing from core to edge)
- Heterogeneous caches c→E (increasing from core to edge)


##  Features

-  **Strict implementation** of original paper equations
-  **5 caching strategies** compared
-  **15 complete simulations** (3 scenarios × 5 strategies)
-  **Zipf distribution** (α=0.8) to model content popularity
-  **Automatic graph generation** (matplotlib)
-  **Detailed metrics** (Server Hits, Hop Reduction, Evictions)
-  **Modular architecture** and extensible
-  **Reproducible results** (fixed seeds)

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Dependencies

```bash
pip install numpy matplotlib
```

### Clone the Repository

```bash
git clone https://github.com/your-username/probcache-simulation.git
cd probcache-simulation
```

##  Usage

### Basic Execution

```bash
python probcache_simulation.py
```

**Output**:
- Detailed logs in terminal
- 6 PNG graphs automatically generated
- Duration: 2-3 minutes

### Example Terminal Output

```
================================================================================
PROBCACHE SIMULATION - Psaras et al., ACM ICN 2012
================================================================================
Topology: Binary tree with 6 levels (63 nodes)
Catalog: 1000 contents
Requests: 100,000
Distribution: Zipf (α=0.8)
================================================================================

SCENARIO 1: HOMOGENEOUS CACHES
--- Cache capacity: 3.0 seconds ---
  Simulation: CE2... ✓ (Server hits: 93.1%)
  Simulation: ProbCache... ✓ (Server hits: 90.0%)

SIMULATION RESULTS
Strategy        Server Hits     Hop Reduction   Total Hops     
--------------------------------------------------------------------------------
CE2                     93.15%         12.99%        986,132
ProbCache               89.98%         14.40%        969,658
```

### Customization

Modify parameters in the `main()` function:

```python
# General configuration
TREE_LEVELS = 6              # Tree depth (6 = 63 nodes)
CATALOG_SIZE = 1000          # Number of distinct contents
NUM_REQUESTS = 100000        # Number of requests
ZIPF_ALPHA = 0.8             # Zipf distribution parameter
```

## 📊 Results

### Comparative Performance (3 seconds capacity)

| Strategy    | Server Hits | Hop Reduction | Cache Hits |
|-------------|-------------|---------------|------------|
| **ProbCache** | **89.98%**  | **14.40%**  | **10.02%** |
| Prob-0.3    | 90.98%      | 13.94%        | 9.02%      |
| CE2         | 93.15%      | 12.99%        | 6.85%      |
| Prob-0.7    | 92.53%      | 13.21%        | 7.47%      |

### Generated Graphs

The simulation automatically generates 6 graphs:

1. **homogeneous_cap_1.0.png** - Minimal capacity (1 sec)
2. **homogeneous_cap_3.0.png** - Reference capacity (3 sec)
3. **homogeneous_cap_5.0.png** - High capacity (5 sec)
4. **heterogeneous_core_to_edge.png** - C→e configuration
5. **heterogeneous_edge_to_core.png** - c→E configuration (optimal)
6. **final_comparison.png** - Synthetic comparison

Each graph contains 4 subplots:
- Server Hit Rate per strategy
- Hop Reduction Ratio per strategy
- Cache Hits per tree level
- Cache Evictions per tree level

### Sample Results Graph



*Figure: Comparison of caching strategies showing ProbCache's superior Hop Reduction performance*

## 🏗️ Architecture

### Main Classes

```python
Content                    # Represents ICN content
LRUCache                   # LRU cache with eviction
Router                     # ICN router with cache
BinaryTree                 # Binary tree topology
CachingStrategy            # Strategy interface
├── CE2Strategy            # Cache Everything Everywhere
├── ProbabilisticStrategy  # Fixed probability
├── ProbCacheStrategy      # Adaptive strategy
└── ProbCachePlusStrategy  # Heterogeneous variant
ICNSimulator               # Simulation engine
```

### Execution Flow

```
1. Create topology (binary tree)
2. Generate requests (Zipf distribution)
3. For each strategy:
   a. Reset caches
   b. Process requests
   c. Calculate metrics
   d. Generate graphs
4. Display comparative results
```

## 📐 Implemented Equations

### ProbCache

The caching probability for a router at position **x**:

```
ProbCache(x) = TimesIn(x) × CacheWeight(x)
```

#### TimesIn(x)

```
TimesIn(x) = (Σ_{i=1}^{c-(x-1)} N_i) / (T_tw × N_x)
```

**Where**:
- `c` = TSI (Time Since Inception) = path length
- `x` = TSB (Time Since Birth) = hops from server
- `N_i` = capacity of router i (in seconds of traffic)
- `N_x` = capacity of current router
- `T_tw` = 10 seconds (time window)

**Interpretation**: TimesIn(x) estimates how many times the content can be served from router x's cache during the time window, considering upstream cache capacity.

#### CacheWeight(x)

```
CacheWeight(x) = x / c
```

**Behavior**:
- Near server (low x) → low probability
- Near client (high x) → high probability

**Rationale**: Favors caching closer to clients to reduce hop count.

### Homogeneous Cache Assumption

For routers with identical capacity `N`:

```
Σ_{i=1}^{c-(x-1)} N_i = N × (c - x + 1)

TimesIn(x) = (c - x + 1) / T_tw
```

### ProbCache+ (Heterogeneous Networks)

For networks with varying cache capacities, ProbCache+ explicitly calculates the sum of upstream capacities by traversing the path toward the server, implementing equations (4) and (5) from the paper.

## 🔧 Extending the Code

### Adding a New Strategy

Create a class inheriting from `CachingStrategy`:

```python
class MyStrategy(CachingStrategy):
    def __init__(self):
        super().__init__("MyStrategy")
    
    def should_cache(self, router: Router, content: Content, 
                     path_length: int, hops_so_far: int) -> bool:
        # Your decision logic
        probability = ...  # Calculate probability
        return random.random() < probability
```

Add it to the tested strategies list:

```python
strategies = [
    CE2Strategy(),
    ProbCacheStrategy(T_tw=10.0),
    MyStrategy()  # Your new strategy
]
```

### Modifying the Topology

Create a new topology class:

```python
class GridTopology:
    def __init__(self, width: int, height: int, capacity: float):
        # Implement 2D grid
        self.width = width
        self.height = height
        # ... build grid ...
```

Replace `BinaryTree` with your topology in `main()`:

```python
network = GridTopology(width=10, height=10, cache_capacity=3.0)
```

### Adding New Metrics

In `ICNSimulator.run_simulation()`, add your metrics:

```python
def run_simulation(self, strategy: CachingStrategy) -> Dict:
    # ... existing code ...
    
    # New metric
    avg_latency = self.total_hops * 10  # 10ms per hop
    
    return {
        # ... existing metrics ...
        'avg_latency_ms': avg_latency
    }
```

##  Testing

### Quick Test (1000 requests)

For rapid testing, modify parameters:

```python
NUM_REQUESTS = 1000      # Instead of 100000
TREE_LEVELS = 4          # Instead of 6
```

Duration: ~10 seconds

### Reproducibility

Results are reproducible thanks to fixed random seeds:

```python
random.seed(42)
np.random.seed(42)
```

Running the simulation multiple times produces identical results.

##  Understanding the Results

### Server Hit Rate
Percentage of requests reaching the origin server. **Lower is better** (indicates effective caching).

### Hop Reduction Ratio
Reduction in the number of hops compared to a network without caching:

```
Hop Reduction = 1 - (Actual Hops / Hops Without Cache)
```

**Higher is better** (indicates bandwidth savings).

### Cache Hit Distribution
Shows how hits are distributed across tree levels. ProbCache concentrates popular content near the root (level 0), with **~32%** of hits at the root level.

### Evictions
Number of times content is removed from cache to make room. Lower evictions indicate better cache utilization.

##  References

### Original Paper

```bibtex
@inproceedings{psaras2012probcache,
  title={Probabilistic in-network caching for information-centric networks},
  author={Psaras, Ioannis and Chai, Wei Koong and Pavlou, George},
  booktitle={Proceedings of the second edition of the ICN workshop on 
             Information-centric networking},
  pages={55--60},
  year={2012},
  organization={ACM},
  doi={10.1145/2342488.2342501}
}
```

### Additional Resources

- [Named Data Networking](https://named-data.net/) - NDN Project
- [ICN Research Group](https://irtf.org/icnrg) - IRTF ICN
- [Content-Centric Networking](https://www.ccnx.org/) - CCNx Project

### Related Work

- **Anand et al., SIGMETRICS 2009**: Redundancy in network traffic
- **Jacobson et al., 2009**: Networking Named Content
- **Chai et al., 2012**: Cache "Less for More" in Information-Centric Networks

##  Known Issues

### LCD Strategy
The Leave Copy Down (LCD) strategy is currently commented out due to potentially incorrect implementation. Results showed:
- Very high server hit rate (99.7%)
- Minimal cache effectiveness (0.31% cache hits)
- Poor hop reduction (9.01%)

This may indicate an implementation bug or a fundamental limitation of LCD in binary tree topologies. Future versions may revisit this implementation.

##  Roadmap

Future enhancements:

- [ ] Fix and re-enable LCD strategy
- [ ] Add support for arbitrary graph topologies
- [ ] Implement variable content sizes
- [ ] Add consumer mobility simulation
- [ ] Support for cache replacement policies (FIFO, LFU)
- [ ] Multi-threaded simulation for large-scale networks
- [ ] Web-based visualization interface
- [ ] Real-world trace replay capability

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

##  Contributing

Contributions are welcome! Please feel free to:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new classes and methods
- Include unit tests for new features
- Update documentation as needed

##  Contact

For questions or suggestions:
- **Pull Requests**: Always welcome!
- **Email**: aissa.kadri@etu.utc.fr

## 🙏 Acknowledgments

- Ioannis Psaras, Wei Koong Chai, and George Pavlou for their paper


## 📊 Citation

If you use this simulation in your research, please cite the original paper:

```bibtex
@inproceedings{psaras2012probcache,
  title={Probabilistic in-network caching for information-centric networks},
  author={Psaras, Ioannis and Chai, Wei Koong and Pavlou, George},
  booktitle={Proceedings of the second edition of the ICN workshop on 
             Information-centric networking},
  pages={55--60},
  year={2012},
  organization={ACM}
}
```

---

** If you find this project useful, please consider giving it a star on GitHub!**

---
