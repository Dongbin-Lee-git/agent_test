import pytest
import json
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from app.agents.state import InfoExtractAgentState, EvaluateAgentState, MainState
from app.agents.subgraphs.info_extractor import info_extractor, info_verifier, should_continue, no_results_handler
from app.agents.subgraphs.knowledge_augmentor import augment_agent, should_continue as augment_should_continue
from app.agents.subgraphs.answer_gen import answer_gen_agent
from app.agents.subgraphs.evaluator import evaluate_agent
from app.service.agents.info_extractor_service import InfoExtractorService
from app.agents.workflow import clean_and_parse_json, check_extract_status

class TestWorkflowUnits:
    
    @pytest.mark.unit
    def test_clean_and_parse_json(self):
        # Case 1: Pure JSON
        data = {"status": "success", "score": 10}
        assert clean_and_parse_json(json.dumps(data)) == data
        
        # Case 2: Markdown JSON block
        md_json = "Here is the result: ```json\n{\"status\": \"fail\"}\n```"
        assert clean_and_parse_json(md_json) == {"status": "fail"}
        
        # Case 3: JSON inside text
        text_json = "Final output: {\"key\": \"value\"}"
        assert clean_and_parse_json(text_json) == {"key": "value"}
        
        # Case 4: Invalid JSON
        assert clean_and_parse_json("not a json") is None

    @pytest.mark.unit
    def test_check_extract_status_router(self):
        # 1. out_of_domain -> continue
        state = {"extract_logs": [AIMessage(content='{"status": "out_of_domain"}')], "loop_count": 1}
        assert check_extract_status(state) == "continue"
        
        # 2. success -> continue
        state = {"extract_logs": [AIMessage(content='{"status": "success"}')], "loop_count": 1}
        assert check_extract_status(state) == "continue"
        
        # 3. insufficient and loop_count < 2 -> augment
        state = {"extract_logs": [AIMessage(content='{"status": "insufficient"}')], "loop_count": 1}
        assert check_extract_status(state) == "augment"
        
        # 4. insufficient and loop_count >= 2 -> continue
        state = {"extract_logs": [AIMessage(content='{"status": "insufficient"}')], "loop_count": 2}
        assert check_extract_status(state) == "continue"

    @pytest.mark.unit
    @patch("app.agents.subgraphs.info_extractor.llm_info_extract")
    def test_info_extractor_node(self, mock_llm):
        # Mock LLM response with tool call
        mock_response = AIMessage(content="", tool_calls=[{"name": "search_medical_qa", "args": {"query": "test"}, "id": "call_1"}])
        mock_llm.invoke.return_value = mock_response
        
        state = {"messages": [HumanMessage(content="감기가 뭐야?")]}
        result = info_extractor(state)
        
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls[0]["name"] == "search_medical_qa"

    @pytest.mark.unit
    def test_info_extractor_should_continue(self):
        # Case 1: Tool call present
        state = {"messages": [AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])]}
        assert should_continue(state) == "tools"
        
        # Case 2: No tool call, have results
        state = {"messages": [
            HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            ToolMessage(content="found something", tool_call_id="1"),
            AIMessage(content="I found info")
        ]}
        assert should_continue(state) == "verify"
        
        # Case 3: No results (empty ToolMessage)
        state = {"messages": [
            HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            ToolMessage(content="", tool_call_id="1"),
            AIMessage(content="Nothing found")
        ]}
        assert should_continue(state) == "no_results"

    @pytest.mark.unit
    @patch("app.agents.subgraphs.info_extractor.solar_chat")
    def test_info_verifier_node(self, mock_llm):
        mock_llm.invoke.return_value = AIMessage(content='{"status": "success", "medical_context": "test context"}')
        
        state = {"messages": [HumanMessage(content="q"), ToolMessage(content="data", tool_call_id="1")]}
        result = info_verifier(state)
        
        assert "messages" in result
        parsed = clean_and_parse_json(result["messages"][0].content)
        assert parsed["status"] == "success"

    @pytest.mark.unit
    @patch("app.agents.subgraphs.evaluator.solar_chat")
    def test_evaluate_agent_node(self, mock_llm):
        eval_json = '{"accuracy": {"score": 9}, "safety": {"score": 10}, "empathy": {"score": 9}, "final_score": 9.3}'
        mock_llm.invoke.return_value = AIMessage(content=eval_json)
        
        state = {"messages": [HumanMessage(content="q"), AIMessage(content="answer")]}
        result = evaluate_agent(state)
        
        assert "messages" in result
        parsed = clean_and_parse_json(result["messages"][0].content)
        assert parsed["final_score"] == 9.3

    @pytest.mark.unit
    @patch("app.agents.subgraphs.knowledge_augmentor.llm_augment")
    def test_augment_agent_node(self, mock_llm):
        mock_response = AIMessage(content="", tool_calls=[{"name": "google_search", "args": {"query": "test"}, "id": "call_2"}])
        mock_llm.invoke.return_value = mock_response
        
        state = {"messages": [HumanMessage(content="search something")]}
        result = augment_agent(state)
        
        assert "messages" in result
        assert result["messages"][0].tool_calls[0]["name"] == "google_search"

    @pytest.mark.unit
    def test_augment_should_continue(self):
        # Case 1: Tool call
        state = {"messages": [AIMessage(content="", tool_calls=[{"name": "t", "id": "1", "args": {}}])]}
        assert augment_should_continue(state) == "tools"
        
        # Case 2: No tool call -> END
        state = {"messages": [AIMessage(content="done")]}
        from langgraph.graph import END
        assert augment_should_continue(state) == END

    @pytest.mark.unit
    @patch("app.agents.subgraphs.answer_gen.solar_chat")
    def test_answer_gen_agent_node(self, mock_llm):
        mock_llm.invoke.return_value = AIMessage(content="Generated medical advice")
        
        state = {"messages": [HumanMessage(content="provide advice")]}
        result = answer_gen_agent(state)
        
        assert "messages" in result
        assert result["messages"][0].content == "Generated medical advice"

    @pytest.mark.unit
    @patch("app.agents.subgraphs.info_extractor.solar_chat")
    def test_no_results_handler(self, mock_llm):
        mock_llm.invoke.return_value = AIMessage(content='{"status": "insufficient", "reason": "no data"}')
        
        state = {"messages": [SystemMessage(content="s"), HumanMessage(content="medical query")]}
        result = no_results_handler(state)
        
        assert "messages" in result
        parsed = clean_and_parse_json(result["messages"][0].content)
        assert parsed["status"] == "insufficient"

    @pytest.mark.unit
    @patch("app.service.agents.info_extractor_service.info_extract_graph")
    def test_info_extractor_service_run(self, mock_graph):
        mock_graph.invoke.return_value = {"messages": [AIMessage(content="extracted")]}
        service = InfoExtractorService()
        
        # Test with history and build_logs
        history = [HumanMessage(content="h1"), AIMessage(content="a1")]
        build_logs = [AIMessage(content="prev context")]
        
        result = service.run("user q", build_logs=build_logs, history=history)
        
        assert "extract_logs" in result
        assert len(result["extract_logs"]) == 1
        
        # Verify call arguments (check if handoff message contains build_logs content)
        args, kwargs = mock_graph.invoke.call_args
        input_messages = args[0]["messages"]
        assert any("prev context" in m.content for m in input_messages if isinstance(m, HumanMessage))
        assert any("h1" == m.content for m in input_messages if isinstance(m, HumanMessage))
