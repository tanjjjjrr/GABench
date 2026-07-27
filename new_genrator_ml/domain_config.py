from typing import Dict, List, Any

CITATION_SEMANTICS = [
    # 1. Path & Flow Algorithms
    {"tool": "shortest_path", "semantics": [
        "the shortest logical chain of knowledge evolution",
        "the most concise citation trajectory between two theoretical concepts",
        "the fastest route for cross-disciplinary concept propagation",
    ]},
    {"tool": "all_shortest_paths", "semantics": [
        "all possible equivalent cognitive pathways between two domains",
        "every optimal derivation sequence from foundational theory to practical application",
    ]},
    {"tool": "maximum_flow", "semantics": [
        "the maximum knowledge throughput between two conflicting schools of thought",
        "the ultimate carrying capacity of information transferred from the source theory to the terminal application",
    ]},
    {"tool": "minimum_cut", "semantics": [
        "the smallest core controversy that could cause a complete disciplinary split",
        "the most critical 'academic barrier' blocking communication between two distinct research directions",
    ]},

    # 2. Structure & Connectivity
    {"tool": "minimum_spanning_tree", "semantics": [
        "the most streamlined core literature skeleton supporting the entire discipline",
        "the essential framework connecting all research milestones with the absolute minimum number of critical citations",
    ]},
    {"tool": "graph_connectivity", "semantics": [
        "evaluating the structural cohesion of the academic community",
        "detecting the emergence of completely isolated 'knowledge islands' in the field",
        "analyzing how many independent, disconnected silos the research field has fragmented into",
    ]},
    # {"tool": "find_cycles", "semantics": [
    #     "detecting the presence of closed-loop, circular reasoning within citation rings",
    #     "identifying self-referential echo chambers within the theoretical framework",
    # ]},
    # {"tool": "topological_sort", "semantics": [
    #     "mapping the strict chronological dependencies of this technology's evolution",
    #     "determining the absolute prerequisite reading sequence required to master this complex domain without cyclical confusion",
    # ]},  # DISABLED: graph contains cycles
    # {"tool": "clique_analysis", "semantics": [
    #     "finding 'academic inner circles' that exclusively endorse each other and co-cite densely",
    #     "identifying the most tightly knit, impenetrable core group of experts in the field",
    # ]},

    # 3. Node Importance & Graph Metrics
    {"tool": "node_centrality", "semantics": [
        "finding the paper universally recognized as the 'main intersection' of the discipline",
        "identifying the keystone publication acting as the primary cross-domain bridge",
        "locating the paper that contributes most to the rapid propagation efficiency of new ideas",
    ]},
    {"tool": "density", "semantics": [
        "evaluating the maximum cognitive breadth (the longest path of knowledge) across the entire research field",
        "measuring the macro-level density and tightness of internal academic discourse",
    ]},
    {"tool": "subgraph_extraction", "semantics": [
        "extracting the localized academic ecosystem surrounding a specific prominent researcher and their direct descendants",
        "isolating the highly specialized vertical research network of a niche subfield",
    ]},

    # 4. Community & Prediction
    {"tool": "community_detection", "semantics": [
        "automatically partitioning the citation network into distinct, self-organizing sub-research clusters",
        "identifying the natural boundaries of emerging interdisciplinary factions",
    ]},
    {"tool": "link_prediction_score", "semantics": [
        "forecasting which two currently unconnected papers are statistically most likely to form a direct citation relationship next",
        "finding the most statistically promising entry points for unprecedented interdisciplinary collaboration",
    ]},
    # 5. Graph Machine Learning & Representation
    {"tool": "structural_node_embedding_classification", "semantics": [
        "determining a paper's true academic discipline solely based on its topological position within the global citation network",
        "inferring the hidden research subfield of an unlabeled publication by analyzing the structural footprint of its surrounding neighborhood",
        "profiling a document's core topic by translating its complex citation environment into a unified conceptual fingerprint"
    ]},
    {"tool": "structural_node_embedding_link_prediction", "semantics": [
        "forecasting the latent probability that Paper A will inevitably cite Paper B based purely on their structural proximity in the network space",
        "identifying a 'missing link' where two structurally compatible papers should conceptually reference each other but currently do not",
        "measuring the topological affinity between two completely disconnected documents to predict future academic cross-pollination"
    ]},
]

RECOMMENDATION_SEMANTICS = [
    # 1. Path & Flow Algorithms
    {"tool": "shortest_path", "semantics": [
        "the most direct cross-selling trajectory between two distinct product categories", 
        "the shortest logical sequence of impulse buys connecting a gateway item to a luxury purchase", 
        "the fastest conversion route for user browsing behavior"
    ]},
    # {"tool": "all_shortest_paths", "semantics": [
    #    "all equivalent consumer browsing patterns leading from an entry-level product to a flagship item", 
    #    "every optimal bundle sequence that maximizes cart value between two specific departments"
    # ]},
    # {"tool": "maximum_flow", "semantics": [
    #    "the maximum potential cross-selling bandwidth between two major product lines", 
    #    "the absolute limit of recommendation traffic that can flow from daily necessities to high-margin electronics"
    # ]},
    # {"tool": "minimum_cut", "semantics": [
    #    "the most critical 'out-of-stock' vulnerability that would completely sever the recommendation pipeline between two departments", 
    #    "the absolute bottleneck in the product catalog that, if removed, stops users from exploring new categories"
    # ]},

    # 2. Structure & Connectivity
    {"tool": "minimum_spanning_tree", "semantics": [
        "the ultimate 'core catalog skeleton' that connects all essential product bundles with the absolute minimum number of co-purchase links", 
        "the most streamlined inventory framework required to maintain cross-category discoverability"
    ]},
    {"tool": "graph_connectivity", "semantics": [
        "evaluating the structural cohesion of the shopping cart ecosystem", 
        "detecting how many isolated 'purchase silos' or independent shopping habits the catalog fragments into", 
        "analyzing the devastating impact of an out-of-stock event on overall catalog discoverability"
    ]},
    # {"tool": "find_cycles", "semantics": [
    #     "detecting closed loops of habitual repurchases or complementary accessory traps", 
    #     "identifying self-contained product ecosystems where users repeatedly buy within the same brand family"
    # ]},
    # {"tool": "topological_sort", "semantics": [
    #     "mapping the strict prerequisite purchasing order for a complex setup (e.g., building a PC or a smart home)",
    #     "determining the absolute logical sequence of upgrades a consumer makes over their customer lifetime"
    # ]},  # DISABLED: graph contains cycles
    # {"tool": "clique_analysis", "semantics": [
    #     "finding 'perfect product bundles' where every single item is frequently bought with every other item in the group", 
    #     "identifying impenetrable micro-niches of highly dedicated hobbyist purchases"
    # ]},

    # 3. Node Importance & Graph Metrics
    # {"tool": "node_centrality", "semantics": [
    #    "finding the 'ultimate gateway product' universally recognized for driving traffic across entirely different departments", 
    #    "identifying the keystone item acting as the primary cross-selling bridge in the recommendation engine", 
    #    "locating the single product that contributes most to exposing users to diverse catalog areas"
    #]},
    #{"tool": "density", "semantics": [
    #    "evaluating the maximum catalog exploration depth (the longest chain of 'frequently bought together' clicks) across the entire store", 
    #    "measuring the macro-level density and tightness of cross-departmental product bundling"
    #]},
    {"tool": "subgraph_extraction", "semantics": [
        "extracting the localized market ecosystem surrounding a specific flagship console and its third-party accessories", 
        "isolating the highly specialized vertical browsing network of a niche lifestyle brand"
    ]},
    # 4. Community & Prediction
    #{"tool": "community_detection", "semantics": [
    #    "automatically partitioning the massive catalog into distinct, organically formed consumer lifestyle clusters", 
    #    "identifying the natural boundaries of hidden consumer demographics based on basket behavior"
    #]},
    {"tool": "link_prediction_score", "semantics": [
        "forecasting which two currently un-associated products are statistically most likely to become the next viral co-purchase trend", 
        "finding the highest-potential latent cross-sell opportunity to optimize the 'Customers who viewed this also viewed' widget"
    ]},
    # 5. Graph Machine Learning & Representation
    {"tool": "structural_node_embedding_classification", "semantics": [
        "determining a product's true category solely based on its topological position within the global recommendation network",
        "inferring the hidden target demographic of an unlabeled product by analyzing the structural footprint of its surrounding co-purchase neighborhood",
        "profiling a product's core segment by translating its complex cross-selling environment into a unified conceptual fingerprint"
    ]},
    {"tool": "structural_node_embedding_link_prediction", "semantics": [
        "forecasting the latent probability that Product A will inevitably be bought with Product B based purely on their structural proximity in the browsing space",
        "identifying a 'missing cross-sell link' where two structurally compatible products should conceptually be bundled together but currently are not",
        "measuring the topological affinity between two completely un-associated products to predict future consumer co-purchase trends"
    ]}
]

SOCIAL_SEMANTICS = [
    # 1. Path & Flow Algorithms (Information Diffusion & Debate)
    {"tool": "shortest_path", "semantics": [
        "the most direct chain of replies connecting a casual user to a platform mega-influencer", 
        "the fastest conversational route for a trend to spread between two distinct online subcultures", 
        "the shortest sequence of user interactions required to bridge two highly polarized users"
    ]},
    # {"tool": "all_shortest_paths", "semantics": [
    #     "all equivalent interaction pathways through which a viral controversy cascaded across the user base", 
    #     "every optimal sequence of debates connecting a niche hobbyist to mainstream platform discourse"
    # ]},
    # {"tool": "maximum_flow", "semantics": [
    #     "the maximum volume of discourse and engagement bandwidth flowing between two opposing ideological echo chambers", 
    #     "the absolute limit of reply traffic that can be sustained between two distinct online communities"
    # ]},
    # {"tool": "minimum_cut", "semantics": [
    #     "the most critical set of 'bridge users' whose account deletion would completely sever communication between two rival groups", 
    #     "the absolute conversational bottleneck that, if removed, fractures a unified debate into isolated echo chambers"
    # ]},

    # 2. Structure & Connectivity (Community Topologies)
    {"tool": "minimum_spanning_tree", "semantics": [
        "the ultimate 'core engagement skeleton' that connects all active platform users with the absolute minimum number of interaction threads", 
        "the most streamlined interaction framework required to keep the overarching platform conversation alive"
    ]},
    {"tool": "graph_connectivity", "semantics": [
        "evaluating the structural cohesion of the platform's daily active user base", 
        "detecting the emergence of completely isolated fringe communities that never interact with outsiders", 
        "analyzing how many independent, disconnected discourse silos the user base has fragmented into"
    ]},
    # {"tool": "find_cycles", "semantics": [
    #     "detecting closed-loop mutual engagement rings or artificial bot networks", 
    #     "identifying self-referential echo chambers where a closed group of users endlessly reply only to each other"
    # ]},
    # {"tool": "topological_sort", "semantics": [
    #     "mapping the strict chronological cascade of who initiated a viral message and who subsequently amplified it",
    #     "determining the absolute chronological dependency of arguments in a massive multi-user debate thread"
    # ]},  # DISABLED: graph contains cycles
    # {"tool": "clique_analysis", "semantics": [
    #     "finding tightly-knit groups where every single user frequently interacts with everyone else in the group", 
    #     "identifying impenetrable micro-communities of highly engaged power users"
    # ]},

    # 3. Node Importance & Graph Metrics (User Influence & Platform Health)
    # {"tool": "node_centrality", "semantics": [
    #     "finding the 'ultimate conversation starter' universally recognized for driving engagement across entirely different communities", 
    #     "identifying the keystone user acting as the primary cultural bridge between disparate groups", 
    #     "locating the single user whose replies contribute most to exposing localized discussions to the global trending page"
    # ]},
    # {"tool": "density", "semantics": [
    #     "evaluating the maximum degrees of separation (diameter) between any two randomly selected users debating on the platform", 
    #     "measuring the macro-level density, hostility, and tightness of cross-community interactions"
    # ]},
    {"tool": "subgraph_extraction", "semantics": [
        "extracting the localized interaction ecosystem surrounding a specific viral public figure and their audience", 
        "isolating the highly specialized vertical debate network of a controversial mega-thread"
    ]},

    # 4. Community & Prediction (Subculture Mining)
    # {"tool": "community_detection", "semantics": [
    #     "automatically partitioning the massive user base into distinct, organically formed subcultures based purely on who they interact with", 
    #     "identifying the hidden, natural boundaries of emerging user demographics without relying on explicit group labels"
    # ]},
    {"tool": "link_prediction_score", "semantics": [
        "forecasting which two currently un-associated users are statistically most likely to get into a direct interaction thread next", 
        "finding the highest-potential latent interaction opportunity to recommend a user to follow or reply to another user"
    ]},

    # 5. Graph Machine Learning & Representation (Social Platform Specific Tasks)
    {"tool": "structural_node_embedding_classification", "semantics": [
        "predicting if a user belongs to the elite 'Top 50% Popular' tier solely based on the structural footprint of who they interact with and the latent topics of their past posts",
        "inferring a user's future platform clout and popularity level by compressing their complex historical community activity and interaction network into a single predictive fingerprint",
        "profiling a user's potential to become a viral influencer by translating their conversational environment and content history into a unified popularity score"
    ]},
    {"tool": "structural_node_embedding_link_prediction", "semantics": [
        "forecasting the latent probability that User A will inevitably interact with User B based purely on the structural proximity of their engagement circles and their historical posting alignment",
        "identifying a 'missing connection' where two users share structurally identical interaction patterns and topical interests, but have never directly interacted",
        "measuring the topological and semantic affinity between two completely disconnected users to predict future debate or agreement in a comment section"
    ]}
]

CITATION_PROMPT = """\
You are an expert AI Benchmark Architect designing natural, human-like tasks for a real-world citation network evaluation.
Your task is to write a single, cohesive request from the perspective of a domain expert asking an AI Agent to perform a complex literature analysis.

Input Variables:
- Dataset Context: {dataset_name}
- First Target Paper: {node1_desc}
- Second Target Paper: {node2_desc}
- Analytical Objectives (Semantics):
{semantics_text}

Constraints:
1. NO Epistolary Format: DO NOT include greetings (e.g., "Dear AI", "Hello"), sign-offs (e.g., "Sincerely"), or conversational pleasantries. The output must be a direct, professional task request.
2. NO Mathematical Placeholders: ABSOLUTELY DO NOT use terms like "Node X", "Node Y", "Node", "Vertex", or "Edge". Integrate {node1_desc} and {node2_desc} directly and naturally into the sentence structure as the actual subjects.
3. Domain-Native Language: Absolutely NO explicit graph theory terminology (e.g., shortest path, max flow). Translate abstract semantics into natural academic/research vocabulary (e.g., instead of "all possible equivalent cognitive pathways," use "trace all independent lineages of thought" or "map out the parallel theoretical developments").
4. Purpose-Driven Integration: Weave the {num_tools} objectives into a cohesive research goal with a realistic motivation (e.g., drafting a literature review). Flow naturally without robotic, sequential phrasing (e.g., DO NOT say "First do this, then calculate that").
5. Minimal Output Instruction: Conclude the request with a simple, generic sentence asking for the results in JSON format. ABSOLUTELY DO NOT reiterate, summarize, or list the specific analytical objectives (e.g., do not write "containing: (a)..., (b)..."). Just ask for the final results to be output in JSON.

Output Format:
Output ONLY the final generated query text. Do not include your reasoning or any introductory filler.
"""

RECOMMENDATION_PROMPT = """\
You are an expert AI Benchmark Architect designing natural, human-like tasks for an E-commerce Recommendation Network evaluation.
Your task is to write a single, cohesive request from the perspective of an E-commerce Strategist (e.g., a Recommender System Engineer, Conversion Rate Optimizer, or Category Manager) asking an AI Agent to perform a complex product network analysis.

Input Variables:
- Dataset Context: {dataset_name}
- Target Product A: {node1_desc}
- Target Product B: {node2_desc}
- Analytical Objectives (Semantics):
{semantics_text}

Constraints:
1. Immersive Recommendation Context: Do NOT use explicit graph theory terminology (e.g., shortest path, connected components). Translate abstract semantics strictly into E-commerce Recommendation terms (e.g., "users who bought this also bought," "clickstream trajectories," "cross-selling funnels," "product co-occurrence clusters," or "conversion bottlenecks").
2. NO Epistolary Format: DO NOT include any greetings (e.g., "Hi Team"), sign-offs (e.g., "Thanks, Retail Team"), or conversational pleasantries. Formulate a direct, professional analytical task brief.
3. NO Mathematical Placeholders: ABSOLUTELY DO NOT use terms like "Node X", "Node Y", "Node", or "Vertex". Integrate {node1_desc} and {node2_desc} directly and organically as the exact items driving the current merchandising or promotional strategy.
4. Purpose-Driven Flow: Weave the {num_tools} objectives into a cohesive business goal (e.g., optimizing a bundle recommendation, analyzing competitive cannibalization, or mapping a user browse journey). Do NOT use robotic, sequential phrasing (e.g., "First do this, then calculate that").
5. Execution Bias: Frame the request so the answer is impossible to guess. It must sound like a complex, data-driven investigation where the Agent is forced to deeply parse the recommendation network to deduce the answer.
6. Uniqueness Guarantee: Append this exact string at the end of the narrative to ensure a single Ground Truth: "(Note: If multiple products yield the exact same structural impact, you MUST select the one with the absolutely smallest Product ID.)"
7. Minimal Output Instruction: Conclude with a single, generic sentence asking for the results in JSON format. If the task involves comparing the two products, mention that the JSON must include 'node_A', 'node_B', and 'is_same'. ABSOLUTELY DO NOT reiterate, summarize, or list the specific analytical objectives or schema keys (e.g., do NOT output lists like "- core_skeleton_min_links..."). 

Output Format:
Output ONLY the final generated query text. Do not include your reasoning or any introductory filler.
"""

SOCIAL_PROMPT = """\
You are an expert AI Benchmark Architect designing natural, human-like tasks for an Online Social Network evaluation. This network is specifically composed of users interacting on an online social software/app.
Your task is to write a single, cohesive request from the perspective of a Platform Administrator or Sociologist asking an AI Agent to perform a complex analysis of this online user social network.

Input Variables:
- Dataset Context: {dataset_name}
- Target User A: {node1_desc}
- Target User B: {node2_desc}
- Analytical Objectives (Semantics):
{semantics_text}

Constraints:
1. Immersive Social Context: Do NOT use explicit graph theory terminology (e.g., shortest path, connected components). Translate abstract semantics strictly into Online Social Network terms (e.g., "viral cascades", "echo chambers", "reply chains", "influencer bridges", or "user communities"). Always maintain the context that these are real users on a social software platform.
2. NO Epistolary Format: DO NOT include any greetings (e.g., "Hi Team"), sign-offs (e.g., "Thanks, Admin"), or conversational pleasantries. Formulate a direct, professional analytical task brief.
3. NO Mathematical Placeholders: ABSOLUTELY DO NOT use terms like "Node X", "Node Y", "Node", or "Vertex". Integrate {node1_desc} and {node2_desc} directly and organically as the exact users driving the current analysis.
4. Purpose-Driven Flow: Weave the {num_tools} objectives into a cohesive analytical goal (e.g., tracking a misinformation campaign, analyzing political polarization, or identifying toxic sub-communities). Do NOT use robotic, sequential phrasing (e.g., "First do this, then calculate that").
5. Execution Bias: Frame the request so the answer is impossible to guess. It must sound like a complex, data-driven investigation where the Agent is forced to deeply parse the user social interaction network to deduce the answer.
6. Uniqueness Guarantee: Append this exact string at the end of the narrative to ensure a single Ground Truth: "(Note: If multiple users or paths yield the exact same structural impact, you MUST select the one with the absolutely smallest User ID/Node ID.)"
7. Minimal Output Instruction: Conclude with a single, generic sentence asking for the results in JSON format. ABSOLUTELY DO NOT reiterate, summarize, or list the specific analytical objectives or schema keys.

Output Format:
Output ONLY the final generated query text. Do not include your reasoning or any introductory filler.
"""

DOMAIN_CONFIGS = {
    "citation": {
        "semantics": CITATION_SEMANTICS,
        "prompt": CITATION_PROMPT,
        "default_dataset_name": "Citation_Graph"
    },
    "recommendation": {
        "semantics": RECOMMENDATION_SEMANTICS,
        "prompt": RECOMMENDATION_PROMPT,
        "default_dataset_name": "E-commerce_Recommendation_Graph"
    },
    "social": {
        "semantics": SOCIAL_SEMANTICS,
        "prompt": SOCIAL_PROMPT,
        "default_dataset_name": "Social_Network_Graph"
    }
}

