import os
import sys
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from manager.manager_graph import Manager_Graph

@pytest.fixture(scope="function", autouse=True)
def graph_manager():
    mg = Manager_Graph()
    mg.delete_entire_graph()
    yield mg
    mg.close()

def test_full_hierarchy_and_retrieval(graph_manager):
    memories = "Thalrik Stormbinder, a name spoken in whispers across the battlefields of Eldrath, emerged from the frostbitten reaches of the Northlands. Born in the village of Varnheim, Thalrik Stormbinder inherited a lineage of warriors and mystics. The blood of the Stormbinders carried a gift—an attunement to the elemental forces that shaped the world. From childhood, the elders of Varnheim recognized an unusual affinity for tempestuous magic in Thalrik Stormbinder, a power both revered and feared among the clans of the Northlands. During the early years, Thalrik Stormbinder studied under the tutelage of Greymane Frostcaller, the eldest shaman of Varnheim. Greymane Frostcaller imparted ancient wisdom concerning the harnessing of wind and lightning, shaping Thalrik Stormbinder’s raw energy into controlled mastery. The training required meditation beneath the storm-laden skies, trials of endurance within the icy waters of Lake Varndar, and communion with the spirits of the mountain peaks. As years passed, the teachings of Greymane Frostcaller transformed into a foundation for the abilities that would later forge a legend. The first great trial of Thalrik Stormbinder arrived during the siege of Varnheim. The warbands of Jorik the Black-Toothed descended upon the village, razing the outer defenses with brutal efficiency. Armed with knowledge and boundless rage, Thalrik Stormbinder stood at the gates, summoning the fury of the skies. The storm answered. Lightning split the battlefield, fire and ice clashed in the howling winds, and the warbands of Jorik the Black-Toothed fled before the tempest’s wrath. The victory over the invaders cemented Thalrik Stormbinder as a warrior-mystic, a guardian of the Northlands. Following the defense of Varnheim, Thalrik Stormbinder sought the lost relics of the Stormbinders, ancient artifacts infused with primordial energy. The journey led across the tundras of Skaldir, through the ruins of Durn’Vok, and into the labyrinthine depths of the Hollow Peak Caves. In the depths of Hollow Peak Caves, the artifact known as the Stormfang Gauntlet was discovered—an armament of gleaming silver, imbued with the raw force of the heavens. With the Stormfang Gauntlet in hand, the command of storms ascended to an unparalleled level, earning Thalrik Stormbinder a place in the annals of the Northlands. The legend of Thalrik Stormbinder reached its zenith during the War of Sundering Tides. The kingdoms of Eldrath waged bitter conflict against the Abyssal Conclave, a cabal of warlocks seeking dominion over the land. The forces of the Abyssal Conclave conjured rifts in the fabric of reality, unleashing horrors from beyond the veil. Kings and warlords faltered before the onslaught, but the banners of the Northlands stood firm under the command of Thalrik Stormbinder. Leading an army of battle-hardened clans, Thalrik Stormbinder struck against the warlocks of the Abyssal Conclave, severing the conduits of abyssal power and shattering the portals that spewed darkness into the world. At the final battle upon the cliffs of Ael’Thun, the culmination of Thalrik Stormbinder’s power reached its peak. The sky churned with obsidian clouds, the sea roared in defiance, and the warlock-lord Malgrith the Hollow prepared the final incantation to engulf Eldrath in eternal night. Thalrik Stormbinder ascended the summit, Stormfang Gauntlet alight with the wrath of the tempest. A single strike, infused with the fury of a thousand storms, tore through the defenses of Malgrith the Hollow, obliterating the abyssal throne and silencing the darkness. The War of Sundering Tides ended with the fall of the Abyssal Conclave. The kingdoms of Eldrath heralded Thalrik Stormbinder as the Stormborn Champion, a title etched into the histories of scholars and bards alike. Despite the accolades, the battle left scars unseen. The power wielded upon the cliffs of Ael’Thun demanded a toll. The once-boundless energy of the storms waned, and the strength once possessed began to fade. Seeking solace, Thalrik Stormbinder departed from the grand cities of Eldrath, returning to the ancestral homeland of Varnheim. In the twilight years, Thalrik Stormbinder passed into legend, vanishing into the frozen expanse beyond the Northern Reach. Some claim that the spirit of the Stormborn Champion roams the high peaks, watching over the Northlands in silence. Others whisper that the legacy of Thalrik Stormbinder remains hidden, waiting for one worthy to claim the mantle of the storm. The tale of the Stormbinder endures, woven into the tapestry of myth, spoken beneath the glow of firelight in the halls of warriors and sages. The chronicles of Thalrik Stormbinder remain a testament to the power of will, the bond between mortals and the elements, and the echoes of a tempest that once shook the foundations of Eldrath. The name Thalrik Stormbinder, spoken in reverence and awe, endures as a symbol of defiance against the tides of darkness, forever bound to the storm that gave birth to a legend."
    graph_manager.text_to_memories(memories, context_window=5)

    community_nodes = graph_manager.run_query("MATCH (c:Community) RETURN c LIMIT 1")
    assert community_nodes is not None and len(community_nodes) > 0, "No Community nodes found in the graph."
    memory_nodes = graph_manager.run_query("MATCH (e:Memory) RETURN e LIMIT 1")
    assert memory_nodes is not None and len(memory_nodes) > 0, "No Memory nodes found in the graph."
    semantic_relationships = graph_manager.run_query("MATCH ()-[r:SEMANTICALLY_RELATED]->() RETURN r LIMIT 1")
    if semantic_relationships:
        assert len(semantic_relationships) > 0, "Semantic relationships should be present but none were found."

    query_text = "Tell me about the battles."
    candidates = graph_manager.retrieve_candidates(query_text, result_limit=10)
    assert isinstance(candidates, list) and len(candidates) > 0, "No candidates retrieved for query."
    found = any("battles" in candidate["content"] for candidate in candidates)
    assert found, "Candidate matching 'battles' was not found."


    print(query_text)

    for candidate in candidates:
        print("\n")
        print(candidate)


if __name__ == "__main__":
    pytest.main()
