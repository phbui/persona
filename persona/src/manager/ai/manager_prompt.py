from meta.meta_singleton import Meta_Singleton
from manager.ai.manager_llm import Manager_LLM

dummy = "Thalrik Stormbinder is a legendary warrior-mystic of the Northlands. Born in Varnheim and trained by the revered shaman Greymane Frostcaller, he mastered the elemental forces of wind and lightning. Fierce and resolute, Thalrik is known for summoning violent storms to defend his homeland—most notably during the siege of Varnheim and the War of Sundering Tides, where he earned the title \"Stormborn Champion.\"Although his formidable power comes at a personal cost, leading him eventually to a quiet withdrawal, his unyielding spirit and decisive reactions in the face of danger remain the essence of his enduring legend."

class Manager_Prompt(metaclass= Meta_Singleton):
    def __init__(self):
        self.manager_llm = Manager_LLM()

    def format_history(self, rounds):
        history = "Chat History:"
        for round in rounds:
            history += f"\n{round['speaker_a']}: {round['message_a']}\n{round['speaker_b']}: {round['message_b']}"

        return history
    
    def format_notes(self, candidates):
        notes = ""
        for candidate in candidates:
            notes += f"\n- {candidate['content']}"

        return notes

    def generate_response(self, rounds, query_data, candidates, agent_name):
        instructions = f"Character Notes: \n- {dummy} "
        notes = self.format_notes(candidates)
        history = self.format_history(rounds) + "\n" + query_data['content']
        name = f"\n{agent_name}:"

        return self.manager_llm.generate_response((instructions + notes + history + name))

    def choose_response(self, rounds, query_data, response_1, response_2):
        instructions = f"Based on the Character Notes and Chat History, which Response (A or B) is character? \n Character Notes\: \n- {dummy} "
        history = self.format_history(rounds) + "\n" + query_data['content']
        responses = f"\n\n Response A: \n {response_1} \n Response B: {response_2} \n\n Answer A or B."
        answer = self.manager_llm.generate_response((instructions + history + responses), max_new_tokens=16, temperature=0.2)

        if "A" in answer.lower():
            return response_1, 1, 0
        else:
            return response_2, 0, 1