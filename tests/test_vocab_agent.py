import sys
import os
# 添加 src 目录到模块搜索路径，以便可以导入 src 目录中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


import unittest
from unittest.mock import patch, MagicMock
from agents.vocab_agent import VocabAgent
from langchain_core.messages import AIMessage

class TestVocabAgent(unittest.TestCase):

    @patch('agents.vocab_agent.AgentBase.load_prompt', return_value='Test prompt')
    def test_vocab_agent_init(self, mock_load_prompt):
        agent = VocabAgent(session_id="vocab_session")
        self.assertEqual(agent.name, "vocab_study")
        self.assertEqual(agent.session_id, "vocab_session")
        self.assertEqual(agent.prompt_file, "prompts/vocab_study_prompt.txt")

    @patch('agents.vocab_agent.get_session_history')
    @patch('utils.logger.LOG.debug')
    def test_restart_session(self, mock_log_debug, mock_get_session_history):
        mock_history = MagicMock()
        mock_get_session_history.return_value = mock_history

        agent = VocabAgent(session_id="vocab_session")
        cleared_history = agent.restart_session(session_id="vocab_session")

        mock_history.clear.assert_called_once()  # Ensure history is cleared
        mock_log_debug.assert_called_once()  # Ensure logging is done
        self.assertEqual(cleared_history, mock_history)  # Check return value

if __name__ == '__main__':
    unittest.main()