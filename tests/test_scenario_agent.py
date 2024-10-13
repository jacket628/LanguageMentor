import sys
import os
# 添加 src 目录到模块搜索路径，以便可以导入 src 目录中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


import unittest
from unittest.mock import patch, MagicMock
import random
from agents.scenario_agent import ScenarioAgent
from langchain_core.messages import AIMessage

class TestScenarioAgent(unittest.TestCase):

    @patch('agents.scenario_agent.AgentBase.load_prompt', return_value='Test prompt')
    @patch('agents.scenario_agent.AgentBase.load_intro', return_value=['Intro message 1', 'Intro message 2'])
    def test_scenario_agent_init(self, mock_load_intro, mock_load_prompt):
        agent = ScenarioAgent(scenario_name="hotel_checkin")
        self.assertEqual(agent.name, "hotel_checkin")
        self.assertEqual(agent.prompt_file, "prompts/hotel_checkin_prompt.txt")
        self.assertEqual(agent.intro_file, "content/intro/hotel_checkin.json")
        self.assertEqual(agent.intro_messages, ['Intro message 1', 'Intro message 2'])

    @patch('agents.scenario_agent.get_session_history')
    @patch('utils.logger.LOG.debug')
    @patch('agents.scenario_agent.random.choice', return_value='Intro message 1')
    def test_start_new_session_without_history(self, mock_random_choice, mock_log_debug, mock_get_session_history):
        mock_history = MagicMock()
        mock_history.messages = []
        mock_get_session_history.return_value = mock_history

        agent = ScenarioAgent(scenario_name="hotel_checkin")
        initial_message = agent.start_new_session(session_id="new_session")

        self.assertEqual(initial_message, 'Intro message 1')
        mock_history.add_message.assert_called_once_with(AIMessage(content='Intro message 1'))

    @patch('agents.scenario_agent.get_session_history')
    def test_start_new_session_with_existing_history(self, mock_get_session_history):
        mock_history = MagicMock()
        mock_history.messages = [AIMessage(content="Existing message")]
        mock_get_session_history.return_value = mock_history

        agent = ScenarioAgent(scenario_name="hotel_checkin")
        last_message = agent.start_new_session(session_id="existing_session")

        self.assertEqual(last_message, "Existing message")
        mock_history.add_message.assert_not_called()

if __name__ == '__main__':
    unittest.main()