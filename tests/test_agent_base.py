import sys
import os
# 添加 src 目录到模块搜索路径，以便可以导入 src 目录中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
from agents.agent_base import AgentBase
from agents.conversation_agent import ConversationAgent
from utils.logger import LOG

class TestAgentBase(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data='This is a prompt.')
    def test_load_prompt_success(self, mock_file):
        agent = AgentBase(name="test_agent", prompt_file="prompts/test_prompt.txt")
        self.assertEqual(agent.prompt, 'This is a prompt.')
        mock_file.assert_called_with("prompts/test_prompt.txt", "r", encoding="utf-8")

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_prompt_file_not_found(self, mock_file):
        with self.assertRaises(FileNotFoundError):
            AgentBase(name="test_agent", prompt_file="missing_prompt.txt")

    @patch('builtins.open', new_callable=mock_open, read_data='["message1", "message2"]')
    def test_load_intro_success(self, mock_file):
        agent = AgentBase(name="test_agent", prompt_file="test_prompt.txt", intro_file="intro.json")
        self.assertEqual(agent.intro_messages, ['message1', 'message2'])

    @patch('builtins.open', side_effect=json.JSONDecodeError("Expecting value", "", 0))
    def test_load_intro_json_decode_error(self, mock_file):
        with self.assertRaises(ValueError):
            AgentBase(name="test_agent", prompt_file="test_prompt.txt", intro_file="invalid_intro.json")

    @patch('agents.session_history.get_session_history')
    @patch('langchain_core.runnables.history.RunnableWithMessageHistory')
    @patch('langchain_ollama.chat_models.ChatOllama')
    @patch('langchain_core.prompts.ChatPromptTemplate.from_messages')
    def test_create_chatbot(self, mock_from_messages, mock_chat_ollama, mock_runnable_with_history,
                            mock_get_session_history):
        mock_prompt_template = MagicMock()
        mock_from_messages.return_value = mock_prompt_template
        agent = AgentBase(name="test_agent", prompt_file="prompts/test_prompt.txt")
        agent.create_chatbot()
        # mock_from_messages.assert_called_once()
        mock_from_messages.assert_called()
        # self.assertTrue(mock_chat_ollama.called)
        # self.assertTrue(mock_runnable_with_history.called)

if __name__ == '__main__':
    unittest.main()